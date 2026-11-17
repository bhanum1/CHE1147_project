# Logistic regression with L1 and L2 regularization — EGFR / erbB1 inhibitors

`logistic_regression_l1_l2.py` trains two logistic-regression classifiers on
`egfr_erbB1/train_zscore.csv`, one with an L1 (lasso) and one with an L2
(ridge) penalty, tunes the regularization strength `C` by cross-validation,
and reports their hold-out performance side by side.

## Data

`egfr_erbB1/train_zscore.csv` — 3 985 molecules × 2 168 columns:

| part | columns | notes |
|---|---|---|
| identifiers | 2 | `chembl_id`, `smiles` — dropped |
| Morgan fingerprints | 2 007 | binary bits, left un-scaled |
| RDKit fragment counts | 66 | `fr_*` |
| RDKit descriptors | 92 | z-scored with train statistics |
| target | 1 | `is_inhibitor` — 2 756 positive / 1 229 negative (69.2 % / 30.8 %) |

No missing values. The 2 165 numeric columns are used as features as-is: the
continuous descriptors were already z-scored against the training set
(`zscore_params.csv`), so re-scaling inside the pipeline would be redundant.

## Usage

```bash
# default: stratified 80/20 hold-out, 5-fold CV, 10 values of C
python logistic_regression_l1_l2.py

# other options
python logistic_regression_l1_l2.py --cv 10 --test-size 0.25
python logistic_regression_l1_l2.py --class-weight balanced
python logistic_regression_l1_l2.py --no-plots
python logistic_regression_l1_l2.py --jobs 1          # serial (no joblib workers)

# score the official test split (refits on 100 % of train_zscore.csv)
python logistic_regression_l1_l2.py --test-data egfr_erbB1/test_zscore.csv
```

Requirements: `scikit-learn`, `pandas`, `numpy`, `matplotlib` (tested with
scikit-learn 1.9, pandas 2.3, numpy 2.4 on Python 3.14).

## How the two models differ

| | L1 (lasso) | L2 (ridge) |
|---|---|---|
| penalty term | `λ·Σ\|w_j\|` | `λ·Σw_j²` |
| solver | `liblinear` | `lbfgs` |
| `C` = 1/λ searched | `1e-3 … 1e2` (10 values) | `1e-3 … 1e2` (10 values) |
| effect on weights | drives many to exactly 0 → feature selection | shrinks all smoothly, keeps all |
| selection criterion | 5-fold stratified CV ROC AUC | same |

`C` is tuned by `GridSearchCV` on ROC AUC because the metric is
threshold-independent and insensitive to the 69/31 class imbalance.

## Results (seed 42, 80/20 split, 797 hold-out molecules)

| penalty | best C | features kept | CV AUC | hold-out AUC | AP | accuracy | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|---|
| L1 | 0.599 | 410 / 2 165 (18.9 %) | 0.8927 ± 0.0102 | 0.8747 | 0.9323 | 0.8206 | 0.8493 | 0.9002 | 0.8740 |
| L2 | 0.167 | 2 159 / 2 165 (99.7 %) | 0.9017 ± 0.0074 | 0.8885 | 0.9429 | 0.8381 | 0.8576 | 0.9183 | 0.8869 |

L2 is ahead by ~0.014 AUC, but L1 reaches nearly the same performance with
**410 features instead of 2 159** — about 5× fewer inputs and a directly
readable list of substructures that drive the prediction.

Both models peak at a moderately strong penalty: with `C = 1e-3` the L1 model
predicts a constant (AUC 0.5, every coefficient shrunk to 0), and with
`C = 100` both start to lose accuracy to overfitting.

## Outputs (`outputs/`)

| file | content |
|---|---|
| `metrics_summary.csv` | all metrics for both models on the evaluation set |
| `cv_results.csv` | mean / std CV AUC for every `C` |
| `sparsity_path_{l1,l2}.csv` | number of non-zero coefficients per `C` |
| `coefficients_l1.csv` | selected features, coefficient, intercept (L1) |
| `coefficients_l2.csv` | all features, coefficient, intercept (L2) |
| `cv_auc_vs_C.png` | CV AUC vs `C`, both penalties |
| `roc_pr_curves.png` | ROC and precision-recall curves, hold-out set |
| `confusion_matrices.png` | confusion matrices at threshold 0.5 |
| `top_coefficients.png` | top-20 features by absolute coefficient, both models |
| `regularization_path.png` | how sparsity changes with `C` |
| `run_log.txt` | console output of the reference run |

## Implementation notes

**The intercept of the L1 model.** `liblinear` appends a synthetic constant
feature whose weight is the intercept, and that weight is itself penalized.
With the default `intercept_scaling=1` a strong L1 penalty drives the
intercept to exactly 0.0 at every `C` — the model then cannot represent the
69/31 class prior and its probabilities are systematically biased (AUC is
unaffected, accuracy and log-loss are). The script sets
`intercept_scaling=100`, which makes the regularizer see the intercept as
~100× cheaper, so it stays effectively unpenalized (intercept −0.90, matching
the ridge model's −0.93) while keeping liblinear's speed. `saga` would handle
the intercept correctly out of the box but needs ~100× longer to converge
here (≈140 s vs ≈0.5 s per fit).

**No leakage.** Every preprocessing-relevant statistic is already baked into
the z-scored CSV, and `C` is tuned only on the training part of the split;
the hold-out set is touched once, for the final score. When `--test-data` is
given, `C` is still selected on `train_zscore.csv` and the models are refit on
all of it before scoring the test split.

**Reproducibility.** `--seed` (default 42) controls the split and the CV
folds; `--jobs 1` forces a serial grid search.
