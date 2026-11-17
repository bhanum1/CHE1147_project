# Verification

- Ten automated tests passed, including fold isolation, missing-value handling, correlation filtering, fixed split membership and serialization for all seven candidates.
- All 48 migrated files retained their original SHA-256 hashes.
- Every candidate has 997 unique reserved-test predictions with finite scores.
- ROC-AUC and MCC were independently recomputed from exported predictions.
- Every serialized model reproduced its stored predictions on a three-molecule sample.
- The selected random forest produced matching scores when SMILES were featurized through the prediction CLI.
- Prediction handling was checked with valid, invalid and missing SMILES.
- No training warnings were captured. A subsequent verification process emitted a harmless Windows physical-core discovery warning and fell back to logical-core counting.
- The CV comparison chart was visually inspected.

The first interrupted preprocessing attempt is explicitly marked incomplete. The complete experiment is 20261111_141725_356816.
