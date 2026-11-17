# Experiment methodology

The proposal's correlation threshold (0.9), excess-kurtosis threshold (7), signed log1p rule and continuous-feature z-scoring are fitted within each CV training fold. Full-data EDA must not be used to choose model parameters. Similar marginal descriptor distributions between classes indicate limited univariate separation; they do not demonstrate normalization quality.

The fixed split manifest is the source of truth. Source label and feature identifiers must be unique and agree. Test IDs never appear in CV. Full-model refits use all training molecules after tuning. Final test metrics do not change the selected model.

Dataset labels are inherited from the Kaggle Cancer Inhibitors dataset (IC50 below 10 micromolar is positive); see data_sources.md. Molecular similarity and assay heterogeneity limit interpretation of random-split results. Scaffold evaluation and feature ablations are future work.
