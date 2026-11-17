# Reports

`eda/` contains the original full-dataset exploratory analysis. `experiments/` contains independent timestamped model runs.

A completed run has `metadata.json`, `cv_summary.csv`, `test_metrics.csv` and `summary.md`. A directory without these files is incomplete; consult its status note or the live training log.

Use the model named by `selected_model` in `metadata.json`. It is selected from training cross-validation, not by sorting test scores. Scores from quick runs or archived experiments are not final results.
