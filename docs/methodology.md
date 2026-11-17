# Experiment methodology

The proposal's correlation threshold (0.9), excess-kurtosis threshold (7), signed log1p rule and continuous-feature z-scoring are fitted within each CV training fold. Full-data EDA must not be used to choose model parameters. Similar marginal descriptor distributions between classes indicate limited univariate separation; they do not demonstrate normalization quality.

The fixed split manifest is the source of truth. Source label and feature identifiers must be unique and agree. Test IDs never appear in CV. Full-model refits use all training molecules after tuning. Final test metrics do not change the selected model.

Dataset labels are inherited from the Kaggle Cancer Inhibitors dataset (IC50 below 10 micromolar is positive); see data_sources.md. Molecular similarity and assay heterogeneity limit interpretation of random-split results.

Scaffold evaluation is implemented. `prepare --scaffold` writes `scaffold_manifest.csv`, assigning whole Bemis-Murcko scaffold groups: the largest scaffold series fill the training set and the tail of rare scaffolds becomes the holdout, so the test set is composed entirely of scaffolds absent from training. Training folds are grouped by scaffold with StratifiedGroupKFold, so tuning cannot lean on a scaffold its own validation fold contains. Acyclic molecules have no Murcko core and are keyed individually rather than merged into one group.

Equally sized scaffold groups are ordered by a seeded shuffle, not alphabetically. Sorting scaffold SMILES by name is not chemically neutral: strings beginning `C(#C` sort first, so an alphabetical tie-break pulls the acetylene-bearing EGFR chemotypes into whichever side fills first. That ordering produced a 5.8-point class-balance gap between the splits; the seeded shuffle reduces it to 1.2 points (69.4% positive in training against 68.2% in test).

Because singleton scaffolds are 23.3% of the dataset and the test target is 20%, the holdout is filled entirely from them: all 694 multi-member scaffolds stay in training, and the test set is 996 molecules with 996 distinct scaffolds. This is an orphan-scaffold holdout, a stricter test than a split that distributes medium-sized series across both sides. Describe it as such rather than as a generic scaffold split.

The two splits answer different questions and both are reported: the random split estimates performance on new molecules from series already seen, the scaffold split estimates transfer to unseen chemistry. Scaffold-split scores are expected to be lower; that gap is a result, not a regression. Feature ablations remain future work.
