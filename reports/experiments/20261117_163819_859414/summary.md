# EGFR experiment

Selected by training CV: **random_forest**.

Split: random. Quick validation: False. Training set: 3985 molecules. Test set: 997 molecules.

Random stratified holdout. Test molecules frequently share a scaffold, and often a near-identical structure, with training molecules, so these scores are an optimistic bound on performance for new chemistry.

Models and hyperparameters are selected by mean cross-validated ROC-AUC. Thresholded metrics use the default decision rule of each model; no operating point is tuned.

```
        model  roc_auc  average_precision  accuracy  precision   recall       f1  balanced_accuracy
     baseline 0.500000           0.692076  0.692076   0.692076 1.000000 0.818020           0.500000
     logistic 0.918784           0.963510  0.847543   0.868493 0.918841 0.892958           0.803068
          knn 0.924649           0.959955  0.854564   0.866756 0.933333 0.898814           0.805429
random_forest 0.933499           0.969789  0.859579   0.876712 0.927536 0.901408           0.817188
```

CV scores were used for tuning and are not unbiased generalization estimates. Test results did not influence model selection. Historical results used different preprocessing and evaluation sets. Coefficients and impurity importance do not establish causal effects or binding.
