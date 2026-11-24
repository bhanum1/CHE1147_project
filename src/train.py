"""Select a model by cross-validation, then score it once on the reserved test set.

The shape of the run, and the reason for it:

1. Every candidate is tuned by grid search over the *same* five frozen folds
   (read from the split manifest, not drawn here). Sharing folds makes the
   comparison between candidates paired, which matters because fold difficulty
   varies more than the gaps between models do.
2. Preprocessing lives inside the pipeline, so the column filters, log decisions
   and scaling are fitted on each training fold alone. Fitting them on the whole
   training set first would leak validation-fold statistics into tuning.
3. The winner is whichever candidate has the best mean CV ROC-AUC. That choice is
   made before any test data is touched, and nothing below can revise it.
4. Only then is every fitted candidate scored on the test set, once.

Everything a run produces lands in one timestamped directory under
`reports/experiments/`, including the library versions and the SHA-256 of both
inputs, so a result can be traced back to exactly what produced it.

Usage (from the repo root):
    python -m egfr_pipeline.train --jobs 4
    python -m egfr_pipeline.train --split scaffold --jobs 4
"""
import argparse, json, hashlib, platform, time, warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import joblib, sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             precision_score, recall_score, f1_score, balanced_accuracy_score,
                             make_scorer, RocCurveDisplay, PrecisionRecallDisplay,
                             ConfusionMatrixDisplay)
from .prepare import ROOT, DATA, main as prepare
from .models import candidates, preprocessing_for
from .preprocessing import MolecularPreprocessor

# ROC-AUC is listed first because `refit='roc_auc'` below makes it the selection
# metric; the rest are recorded for description only. `precision` needs an
# explicit scorer so that a degenerate fold scores 0 instead of raising.
SCORING = {
    'roc_auc': 'roc_auc',
    'average_precision': 'average_precision',
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, zero_division=0),
    'recall': 'recall',
    'f1': 'f1',
    'balanced_accuracy': 'balanced_accuracy',
}

SPLIT_NOTES = {
    'random': 'Random stratified holdout. Test molecules frequently share a scaffold, and often '
              'a near-identical structure, with training molecules, so these scores are an '
              'optimistic bound on performance for new chemistry.',
    'scaffold': 'Bemis-Murcko scaffold holdout. No test scaffold appears in training or in any '
                'tuning fold, so these scores estimate transfer to unseen chemistry and are '
                'expected to be lower than random-split scores.',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', help='Subset of candidates to run (default: all)')
    parser.add_argument('--jobs', type=int, default=4, help='Parallel workers for the grid search')
    parser.add_argument('--quick', action='store_true',
                        help='Keep the folds but shrink every grid to one configuration')
    parser.add_argument('--config', type=Path, help='JSON file of replacement parameter grids')
    parser.add_argument('--split', choices=['random', 'scaffold'], default='random',
                        help='Evaluation split: the original random holdout, or the Bemis-Murcko scaffold holdout')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be a positive integer')
    unknown = set(args.models or []) - set(candidates())
    if unknown:
        parser.error('Unknown model names: ' + ', '.join(sorted(unknown)))

    # --- Inputs -------------------------------------------------------------
    manifest_path = DATA / ('split_manifest.csv' if args.split == 'random' else 'scaffold_manifest.csv')
    if not manifest_path.exists():
        prepare([] if args.split == 'random' else ['--scaffold'])
    manifest = pd.read_csv(manifest_path)
    data = pd.read_csv(DATA / 'features.csv').set_index('chembl_id')

    tr = manifest[manifest.split == 'train']
    te = manifest[manifest.split == 'test']
    train = data.loc[tr.chembl_id]
    test = data.loc[te.chembl_id]
    X = train.drop(columns=['smiles', 'is_inhibitor'])
    y = train.is_inhibitor
    Xt = test[X.columns]
    yt = test.is_inhibitor

    # The folds come from the manifest as explicit index pairs rather than from a
    # CV splitter, so every candidate sees byte-identical partitions.
    fold_ids = tr.fold.to_numpy()
    folds = [(np.flatnonzero(fold_ids != i), np.flatnonzero(fold_ids == i)) for i in range(5)]

    run = (datetime.now().strftime('%Y%m%d_%H%M%S_%f')
           + ('_scaffold' if args.split == 'scaffold' else '')
           + ('_quick' if args.quick else ''))
    out = ROOT / 'reports/experiments' / run
    out.mkdir(parents=True)
    modeldir = ROOT / 'models' / run
    modeldir.mkdir(parents=True)
    config = json.loads(args.config.read_text(encoding='utf-8-sig')) if args.config else {}

    # --- Model selection ----------------------------------------------------
    searches = {}
    rows = []
    messages = []
    for name, (estimator, grid) in candidates().items():
        if args.models and name not in args.models:
            continue
        grid = config.get(name, grid)
        if args.quick:
            grid = {k: [v[0]] for k, v in grid.items()}

        # `memory` caches the fitted preprocessor per fold, so the expensive
        # correlation filter runs once per fold instead of once per configuration.
        pipe = Pipeline([('preprocess', MolecularPreprocessor(**preprocessing_for(name))),
                         ('model', estimator)],
                        memory=str(ROOT / '.cache/preprocessing'))
        search = GridSearchCV(pipe, {'model__' + k: v for k, v in grid.items()},
                              scoring=SCORING, refit='roc_auc', cv=folds,
                              n_jobs=args.jobs, error_score='raise', verbose=1)

        start = time.monotonic()
        print(f'Training {name}...', flush=True)
        # Warnings are captured rather than printed: convergence and
        # zero-division notices are worth keeping with the run, not scrolling past.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            search.fit(X, y)
        messages.extend(f'{name}: {w.message}' for w in caught)

        results = pd.DataFrame(search.cv_results_)
        results.to_csv(out / f'{name}_cv.csv', index=False)
        row = {'model': name, 'seconds': time.monotonic() - start,
               'best_params': json.dumps(search.best_params_)}
        for metric in SCORING:
            row[f'cv_{metric}'] = results.loc[search.best_index_, f'mean_test_{metric}']
            row[f'cv_{metric}_std'] = results.loc[search.best_index_, f'std_test_{metric}']
        rows.append(row)
        searches[name] = search

        joblib.dump(search.best_estimator_, modeldir / f'{name}.joblib')

        # Record what preprocessing actually decided, so a dropped column or an
        # unexpected log transform can be traced after the fact.
        pre = search.best_estimator_.named_steps['preprocess']
        (out / f'{name}_preprocessing.json').write_text(json.dumps({
            'retained': pre.columns_, 'dropped': pre.dropped_, 'logged': pre.logged_,
            'means': pre.means_.to_dict(), 'scales': pre.scales_.to_dict(),
            'medians': pre.medians_.to_dict()}, indent=2))

        # Coefficients for the linear model, impurity importance for the forest;
        # KNN and the baseline have neither.
        model = search.best_estimator_.named_steps['model']
        values = model.coef_.ravel() if hasattr(model, 'coef_') else getattr(model, 'feature_importances_', None)
        if values is not None:
            pd.DataFrame({'feature': pre.columns_, 'value': values}).to_csv(
                out / f'{name}_importance.csv', index=False)

        print(f'{name}: CV AUC={row["cv_roc_auc"]:.4f}, elapsed={row["seconds"]:.1f}s', flush=True)

    if not rows:
        raise ValueError('No models selected')

    cv = pd.DataFrame(rows).sort_values(['cv_roc_auc', 'cv_f1', 'model'],
                                        ascending=[False, False, True])
    cv.to_csv(out / 'cv_summary.csv', index=False)
    winner = cv.iloc[0].model

    # --- Test evaluation ----------------------------------------------------
    # Selection is frozen above. Nothing past this point can change `winner`;
    # the test set is read once, for description.
    final = []
    figroc, axroc = plt.subplots()
    figpr, axpr = plt.subplots()
    for name, search in searches.items():
        model = search.best_estimator_

        # Ranking metrics need a continuous score. Logistic regression exposes a
        # signed margin, the others a probability; which one was used is recorded
        # alongside the predictions so the column is never misread as a probability.
        kind = 'decision_score' if hasattr(model, 'decision_function') else 'probability'
        score = model.decision_function(Xt) if kind == 'decision_score' else model.predict_proba(Xt)[:, 1]
        predicted = model.predict(Xt)

        final.append({'model': name,
                      'roc_auc': roc_auc_score(yt, score),
                      'average_precision': average_precision_score(yt, score),
                      'accuracy': accuracy_score(yt, predicted),
                      'precision': precision_score(yt, predicted, zero_division=0),
                      'recall': recall_score(yt, predicted),
                      'f1': f1_score(yt, predicted),
                      'balanced_accuracy': balanced_accuracy_score(yt, predicted)})

        # Per-molecule predictions are saved so figures and paired significance
        # tests can be rebuilt later without refitting anything.
        pd.DataFrame({'chembl_id': te.chembl_id.to_numpy(), 'label': yt.to_numpy(),
                      'prediction': predicted, 'score': score, 'score_type': kind}
                     ).to_csv(out / f'{name}_predictions.csv', index=False)

        RocCurveDisplay.from_predictions(yt, score, name=name, ax=axroc)
        PrecisionRecallDisplay.from_predictions(yt, score, name=name, ax=axpr)
        display = ConfusionMatrixDisplay.from_predictions(yt, predicted)
        display.ax_.set_title(name)
        display.figure_.savefig(out / f'{name}_confusion.png')
        plt.close(display.figure_)

    figroc.savefig(out / 'roc.png', bbox_inches='tight')
    figpr.savefig(out / 'precision_recall.png', bbox_inches='tight')
    plt.close('all')

    final = pd.DataFrame(final)
    final.to_csv(out / 'test_metrics.csv', index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(cv.model, cv.cv_roc_auc, yerr=cv.cv_roc_auc_std)
    ax.tick_params(axis='x', rotation=40)
    ax.set_ylabel('CV ROC-AUC')
    fig.tight_layout()
    fig.savefig(out / 'cv_comparison.png')
    plt.close(fig)

    # --- Provenance ---------------------------------------------------------
    # The two hashes are the point of this block: they pin the result to one
    # feature table and one split manifest.
    metadata = {'selected_model': winner, 'split': args.split,
                'n_train': len(tr), 'n_test': len(te), 'quick': args.quick, 'seed': 42,
                'python': platform.python_version(), 'sklearn': sklearn.__version__,
                'numpy': np.__version__, 'pandas': pd.__version__,
                'arguments': vars(args), 'configuration': config,
                'feature_sha256': hashlib.sha256((DATA / 'features.csv').read_bytes()).hexdigest(),
                'split_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2, default=str))
    (out / 'warnings.txt').write_text('\n'.join(messages), encoding='utf-8')

    (out / 'summary.md').write_text(
        f'# EGFR experiment\n\nSelected by training CV: **{winner}**.\n\n'
        f'Split: {args.split}. Quick validation: {args.quick}. Training set: {len(tr)} molecules. '
        f'Test set: {len(te)} molecules.\n\n{SPLIT_NOTES[args.split]}\n\n'
        'Models and hyperparameters are selected by mean cross-validated ROC-AUC. Thresholded '
        'metrics use the default decision rule of each model; no operating point is tuned.\n\n'
        f'```\n{final.to_string(index=False)}\n```\n\n'
        'CV scores were used for tuning and are not unbiased generalization estimates. Test '
        'results did not influence model selection. Historical results used different '
        'preprocessing and evaluation sets. Coefficients and impurity importance do not '
        'establish causal effects or binding.\n', encoding='utf-8')

    print(f'Completed: {out}', flush=True)


if __name__ == '__main__':
    main()
