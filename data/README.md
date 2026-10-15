# Protein kinase inhibitor data

There is one CSV per protein kinase. Each CSV has two columns:

- `chembl_id`: the molecule's ChEMBL ID
- `is_inhibitor`: 1 if the molecule's IC50 is below 10 µM, otherwise 0

The source is the [Kaggle Cancer Inhibitors dataset](https://www.kaggle.com/datasets/xiaotawkaggle/inhibitors). The IDs and labels come from the dataset's `.h5` files, because its CSVs contain fingerprints only and have no IDs.

## Duplicates

The raw data lists some molecules more than once. When all copies of a molecule have the same label, one row is kept. When the copies disagree, the molecule is dropped because its label is unclear.

| Protein | Raw rows | Clean | Inhibitors | Non-inhibitors | Dropped (conflicting) |
|---|---|---|---|---|---|
| cdk2 | 1890 | 1634 | 1039 | 595 | 2 |
| egfr_erbB1 | 6937 | 4991 | 3454 | 1537 | 187 |
| gsk3b | 2208 | 1932 | 1430 | 502 | 20 |
| hgfr | 2879 | 2134 | 1824 | 310 | 11 |
| map_k_p38a | 4670 | 3521 | 3051 | 470 | 76 |
| tpk_lck | 2169 | 1801 | 1297 | 504 | 8 |
| tpk_src | 3587 | 2947 | 1878 | 1069 | 31 |
| vegfr2 | 6860 | 5085 | 4090 | 995 | 68 |
