# EGFR experiment

Selected by training CV: **random_forest**.

Quick validation: False. Test set: 997 molecules.

```
        model  roc_auc  average_precision  accuracy  precision   recall       f1      mcc  balanced_accuracy
     baseline 0.500000           0.692076  0.692076   0.692076 1.000000 0.818020 0.000000           0.500000
  logistic_l1 0.911212           0.959054  0.846540   0.921507 0.850725 0.884702 0.661107           0.843929
  logistic_l2 0.916235           0.961651  0.849549   0.879213 0.907246 0.893010 0.640736           0.813558
          knn 0.925622           0.962245  0.854564   0.863818 0.937681 0.899236 0.646537           0.802717
random_forest 0.933671           0.969872  0.862588   0.879287 0.928986 0.903453 0.668849           0.821170
   svm_linear 0.917509           0.960161  0.852558   0.873453 0.920290 0.896260 0.644714           0.810308
      svm_rbf 0.932403           0.968218  0.873621   0.894958 0.926087 0.910256 0.698072           0.840894
```

CV scores were used for tuning and are not unbiased generalization estimates. Test results did not influence model selection. Historical results used different preprocessing and evaluation sets. Coefficients and impurity importance do not establish causal effects or binding.
