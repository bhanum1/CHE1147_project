# EGFR inhibitor classification

This CHE1147 project compares molecular classifiers for EGFR inhibition. Other kinase label tables are retained as source material, but the current experiments use EGFR only.

## Completed experiment

Four candidates over 51 hyperparameter configurations and 255 fits per split. Random forest was selected on both, with the same settings each time: unlimited depth, `max_features='sqrt'`, `min_samples_leaf=1`. Logistic regression chose ridge (`l1_ratio=0`) on both, and Tanimoto KNN chose distance weighting.

ROC-AUC is the primary metric and the basis for selection. Thresholded metrics use the default decision rule of each candidate; no operating point is tuned.

| | CV ROC-AUC | test ROC-AUC | avg precision | F1 | precision | recall | balanced acc |
|---|---|---|---|---|---|---|---|
| random split | | | | | | | |
| random_forest | 0.9271 | 0.9335 | 0.9698 | 0.9014 | 0.877 | 0.928 | 0.8172 |
| knn (Tanimoto) | 0.9089 | 0.9246 | 0.9600 | 0.8988 | 0.867 | 0.933 | 0.8054 |
| logistic (C=0.3, ridge) | 0.9077 | 0.9188 | 0.9635 | 0.8930 | 0.868 | 0.919 | 0.8031 |
| baseline | 0.5000 | 0.5000 | 0.6921 | 0.8180 | 0.692 | 1.000 | 0.5000 |
| [scaffold split](reports/experiments/20261117_164744_647852_scaffold/summary.md) | | | | | | | |
| random_forest | 0.8798 | 0.8744 | 0.9384 | 0.8548 | 0.786 | 0.937 | 0.6955 |
| knn (Tanimoto) | 0.8495 | 0.8586 | 0.9231 | 0.8502 | 0.785 | 0.928 | 0.6910 |
| logistic (C=0.1, ridge) | 0.8537 | 0.8523 | 0.9238 | 0.8525 | 0.815 | 0.894 | 0.7293 |
| baseline | 0.5000 | 0.5000 | 0.6817 | 0.8107 | 0.682 | 1.000 | 0.5000 |

Read the scaffold numbers as the estimate of transfer to new chemistry and the random-split numbers as an optimistic same-series bound. To view the original random-split summary from the repository root, run:

```bash
git show 332819d:reports/experiments/20261117_163819_859414/summary.md
```

Random forest beats both baselines, and the comparison has to be paired to see it: fold difficulty varies enough that marginal error bars overlap while within-fold differences do not. The forest wins all five folds against both knn and logistic on both splits, by mean per-fold margins of 0.026 to 0.030 on the scaffold split. A paired bootstrap over the scaffold test molecules puts the advantage at +0.0158 ROC-AUC against knn (95% CI +0.0007 to +0.0308) and +0.0221 against logistic (95% CI +0.0045 to +0.0399). Two caveats: cross-validation folds share training data, so paired t-tests across them are anti-conservative, and the margin over knn is narrow enough that it would not survive a correction for multiple comparisons. The margin over the majority-class baseline, +0.3744, is not in doubt.

On the random split the same comparison against knn is not significant (95% CI -0.0027 to +0.0208). A Tanimoto lookup reaches 0.9246 there, within 0.009 of the tuned ensemble, which measures directly how little that split asks of a model.

Hyperparameter headroom is spent: all 18 random forest configurations land within one fold standard deviation of the best on the scaffold split, spanning 0.0115 ROC-AUC in total.

The serialized selected pipelines are under `models/<run>/` (generated locally and ignored by Git).

## Setup

Requires Python 3.11 or newer. Create a virtual environment, install the dependencies, then install this package:

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
```

```bash
python -m pip install -r requirements.txt
```

```bash
python -m pip install -e .
```

`requirements.txt` gives minimum tested versions rather than exact pins; the exact versions behind a recorded experiment are in that run's `metadata.json`.

## Run

Run these commands from the repository root.

```bash
python -m src.prepare
python -m src.prepare --scaffold
python -m src.prepare --rebuild-features
python -m src.eda egfr_erbB1
python -m src.train --jobs 4
python -m src.train --split scaffold --jobs 4
python -m src.train --quick --jobs 1
python -m src.train --models logistic random_forest --jobs 4
python -m src.predict --model models/RUN/random_forest.joblib --input molecules.csv --output predictions.csv
```

Prediction input requires a `smiles` column. Invalid structures are retained with an error status. `--quick` reduces the parameter grid while retaining the five fixed folds; its results are validation artifacts, not final experiment results. `--config configs/search.json` can supply model-specific parameter grids.

`prepare` with no arguments verifies the random split against the feature table and rewrites its five folds; the 3985/997 membership itself is historical and is kept in `split_manifest.csv`, which cannot be regenerated. `--scaffold` generates the scaffold split from scratch. Preparation normally reuses the checked-in complete feature table. `--rebuild-features` streams the local ChEMBL dictionary and writes `features_rebuilt.csv` separately, checking molecule membership before saving; it does not replace the experiment input automatically. Optional `ydata-profiling` enables a fresh HTML report; core EDA runs without it.

For predictable CPU usage, set `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, and `MKL_NUM_THREADS` to `1` before launching the parallel search, so the BLAS threads do not oversubscribe the cores already used by `--jobs`. The recorded full run uses four search workers.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m src.train --jobs 4
```

On Windows PowerShell, set the same three variables with `$env:OPENBLAS_NUM_THREADS = "1"` before running the command.

## Layout

- `data/raw/egfr_erbB1.csv`: cleaned source labels.
- `data/external/`: local ChEMBL structure dictionary, needed only by `--rebuild-features` and too large to check in.
- `data/processed/egfr_erbB1/`: 2265 unfiltered molecular features, the fixed random split membership and the scaffold split membership.
- `src/`: preprocessing, candidate models, preparation, exploratory analysis, training and prediction.
- `reports/eda/`: exploratory analysis; this describes the complete dataset and is not a model-selection input.
- `reports/experiments/`: timestamped CV and final test results, figures and metadata.
- `report/`: the figures used in the final report, and the scripts that build them.
- `models/`: serialized complete pipelines, generated locally and ignored by Git.

## Evaluation

Two splits are available. `--split random` (the default) preserves the original 3985/997 random stratified membership with five-fold stratified training CV. `--split scaffold` holds out whole Bemis-Murcko scaffolds, 3986/996, and groups the five training folds by scaffold as well, so no scaffold is ever shared between training and either a validation fold or the test set.

The random split is heavily redundant: 5.5% of its test molecules have a training molecule with an identical Morgan fingerprint, 44.7% have one at Tanimoto 0.8 or above, and simply copying the nearest training neighbour's label scores 85.8% test accuracy against the selected random forest's 86.3%. Its scores are therefore an optimistic bound that largely measures retrieval of near-duplicates. The scaffold split cuts exact-twin overlap to 0.4% and nearest-neighbour label copying to 78.5%. Report both, and read the random-split numbers as same-series interpolation rather than transfer to new chemistry.

Every model uses identical five-fold training CV (seed 42). Missing-column selection, variance and correlation filtering, log-transform decisions and scaling are fitted inside each training fold. Unexpected missing values at prediction time use training medians. Binary features remain binary. Random forests omit log transforms and scaling.

Four candidates, in increasing order of capacity: a majority-class baseline, regularized logistic regression, a Tanimoto nearest-neighbour baseline and a random forest. The logistic penalty is a hyperparameter, not a separate model: `l1_ratio` 0 is ridge and 1 is lasso, chosen by cross-validation. KNN uses Jaccard distance, which is 1 - Tanimoto on binary fingerprints, and is the only candidate fitted on the fingerprint alone, without descriptors; it is the read-across baseline the learned models have to beat.

Models and hyperparameters are selected by mean CV ROC-AUC. All fitted candidates are evaluated on the reserved test set only after selection is frozen. CV estimates used during tuning are not unbiased estimates of generalization.

Metrics include ROC-AUC, average precision, accuracy, precision, recall, F1 and balanced accuracy. ROC-AUC is primary; it is what selection uses and what the comparisons between models rest on. The thresholded metrics describe the default operating point only. F1 in particular ignores true negatives and rewards predicting the majority class, so the all-positive baseline already scores above 0.81 here; read it against that floor rather than against zero, and alongside precision and recall. Coefficients and impurity feature importance describe model associations, not causal chemistry or molecular binding.

Historical scores from the earlier coursework came from a 797-molecule internal holdout with preprocessing fitted outside CV, and must not be compared directly with the evaluation above.
