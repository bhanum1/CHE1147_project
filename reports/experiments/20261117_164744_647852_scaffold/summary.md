# EGFR experiment

Selected by training CV: **random_forest**.

Split: scaffold. Quick validation: False. Training set: 3986 molecules. Test set: 996 molecules.

Bemis-Murcko scaffold holdout. No test scaffold appears in training or in any tuning fold, so these scores estimate transfer to unseen chemistry and are expected to be lower than random-split scores.

Models and hyperparameters are selected by mean cross-validated ROC-AUC. Thresholded metrics use the default decision rule of each model; no operating point is tuned.

```
        model  roc_auc  average_precision  accuracy  precision   recall       f1  balanced_accuracy
     baseline 0.500000           0.681727  0.681727   0.681727 1.000000 0.810746           0.500000
     logistic 0.852283           0.923798  0.789157   0.814765 0.893962 0.852528           0.729315
          knn 0.858648           0.923138  0.777108   0.784558 0.927835 0.850202           0.691047
random_forest 0.874430           0.938395  0.783133   0.786156 0.936672 0.854839           0.695465
```

CV scores were used for tuning and are not unbiased generalization estimates. Test results did not influence model selection. Historical results used different preprocessing and evaluation sets. Coefficients and impurity importance do not establish causal effects or binding.
