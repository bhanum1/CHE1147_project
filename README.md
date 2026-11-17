# EGFR inhibitor classification

This CHE1147 project compares molecular classifiers for EGFR inhibition. Other kinase label tables are retained as source material, but the current experiments use EGFR only.

## Completed experiment

The full five-fold search is available in [the experiment summary](reports/experiments/20261111_141725_356816/summary.md). Random forest was selected by training CV ROC-AUC (0.9271); its reserved-test ROC-AUC is 0.9337, F1 is 0.9035 and MCC is 0.6688. RBF SVM has higher test F1/MCC, but test metrics do not change the model selected during training. See the experiment directory for all seven candidates, figures, configurations and logs.

The serialized selected pipeline is `models/20261111_141725_356816/random_forest.joblib` (generated locally and ignored by Git). Ten automated tests pass; migrated files retain their original content hashes.

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

Without installing the package, put `src` on the import path instead: `export PYTHONPATH="$PWD/src"` (Windows PowerShell: `$env:PYTHONPATH = "$PWD/src"`).

## Run

```bash
python -m egfr_pipeline.prepare
python -m egfr_pipeline.prepare --rebuild-features
python -m egfr_pipeline.eda egfr_erbB1
python -m pytest -q
python -m egfr_pipeline.train --jobs 4
python -m egfr_pipeline.train --quick --jobs 1
python -m egfr_pipeline.train --models logistic_l1 logistic_l2 --jobs 4
python -m egfr_pipeline.predict --model models/RUN/logistic_l2.joblib --input molecules.csv --output predictions.csv
```

Prediction input requires a `smiles` column. Invalid structures are retained with an error status. `--quick` reduces the parameter grid while retaining the five fixed folds; its results are validation artifacts, not final experiment results. `--config configs/search.json` can supply model-specific parameter grids.

Preparation normally reuses the checked-in complete feature table. `--rebuild-features` streams the local ChEMBL dictionary and writes `features_rebuilt.csv` separately, checking molecule membership before saving; it does not replace the experiment input automatically. Optional `ydata-profiling` enables a fresh HTML report; core EDA runs without it.

For predictable CPU usage, set `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, and `MKL_NUM_THREADS` to `1` before launching the parallel search, so the BLAS threads do not oversubscribe the cores already used by `--jobs`. The recorded full run uses four search workers.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m egfr_pipeline.train --jobs 4
```

On Windows PowerShell, set the same three variables with `$env:OPENBLAS_NUM_THREADS = "1"` before running the command.

## Layout

- `docs/`: proposal, data provenance, methodology and a SHA-256 migration manifest.
- `data/raw/`: cleaned source labels for eight kinases.
- `data/external/`: local ChEMBL structure dictionary.
- `data/interim/`: EGFR structures and initial descriptors.
- `data/processed/egfr_erbB1/`: 2265 unfiltered molecular features and fixed split membership.
- `src/egfr_pipeline/`: preprocessing, candidate models, preparation, training and prediction.
- `reports/eda/`: original exploratory analysis; this describes the complete dataset and is not a model-selection input.
- `reports/experiments/`: timestamped CV and final test results, figures and metadata.
- `models/`: serialized complete pipelines, generated locally.
- `legacy/`: historical scripts and outputs, retained for traceability and not used as training features.

## Evaluation

The original 3985/997 training/test membership is preserved. Every model uses identical five-fold stratified training CV (seed 42). Missing-column selection, variance and correlation filtering, log-transform decisions and scaling are fitted inside each training fold. Unexpected missing values at prediction time use training medians. Binary features remain binary. Random forests omit log transforms and scaling.

Candidates are majority baseline, L1/L2 logistic regression, KNN, random forest and linear/RBF SVM. Models and hyperparameters are selected by mean CV ROC-AUC. All fitted candidates are evaluated on the reserved test set only after selection is frozen. CV estimates used during tuning are not unbiased estimates of generalization. SVM outputs are decision scores, not probabilities. Default model thresholds are used, without test-based threshold tuning.

Metrics include ROC-AUC, average precision, accuracy, precision, recall, F1, MCC and balanced accuracy. Coefficients and impurity feature importance describe model associations, not causal chemistry or molecular binding.

Historical scores came from a 797-molecule internal holdout with preprocessing outside CV and must not be compared directly with the corrected evaluation. The partial CDK2 feature artifact has only 180 molecules and no target column.

Original scripts in `legacy/` retain their historical path assumptions and are reference-only. Use the package entrypoints for current work.
