# Reports

`eda/` contains the original full-dataset exploratory analysis. `experiments/` contains independent timestamped model runs.

A completed run has `metadata.json`, `cv_summary.csv`, `test_metrics.csv` and `summary.md`. A directory without these files is incomplete; consult its status note or the live training log.

`metadata.json` records which split the run used. Directories suffixed `_scaffold` hold out whole Bemis-Murcko scaffolds; unsuffixed directories use the original random split, whose scores are inflated by near-duplicate molecules shared between training and test. Do not compare scores across the two.

Use the model named by `selected_model` in `metadata.json`. It is selected from training cross-validation, not by sorting test scores. Scores from quick runs or archived experiments are not final results.
