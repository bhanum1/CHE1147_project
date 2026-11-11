# Historical experiments

These files are preserved reference material. They retain their original contents and path assumptions; they are not the current executable workflow.

- `splits/`: previous filtered/scaled tables, logistic regression implementation and preliminary outputs. The split CSVs supply ID membership only to the new preparation command.
- `eda.py`, `featurize.py`, `split.py`, `normalize.py`: original preprocessing and analysis scripts.
- `gemini-code-*.py`: exploratory structure retrieval scripts.
- `cdk2_features.csv`: partial 180-molecule artifact without labels.

Use `python -m egfr_pipeline.train` for corrected fold-local preprocessing and model comparisons. See `docs/migration.json` for original locations and verified content hashes.
