# CHE1147 Project: Protein Kinase Inhibitor Classification

Course project for CHE1147. The task is to predict, from molecular structure, whether a small molecule inhibits each of 8 protein kinases.

| Abbreviation | Protein |
|---|---|
| cdk2 | Cyclin-dependent kinase 2 |
| egfr_erbB1 | Epidermal growth factor receptor erbB1 |
| gsk3b | Glycogen synthase kinase-3 beta |
| hgfr | Hepatocyte growth factor receptor |
| map_k_p38a | MAP kinase p38 alpha |
| tpk_lck | Tyrosine-protein kinase LCK |
| tpk_src | Tyrosine-protein kinase SRC |
| vegfr2 | Vascular endothelial growth factor receptor 2 |

## Data

`data/` has one CSV per protein, with columns `chembl_id` and `is_inhibitor`. A molecule counts as an inhibitor if its IC50 in ChEMBL is below 10 µM. See [data/README.md](data/README.md) for where the data came from and how duplicates were handled.

The source is the [Kaggle Cancer Inhibitors dataset](https://www.kaggle.com/datasets/xiaotawkaggle/inhibitors) (CC BY-NC-SA 4.0).
