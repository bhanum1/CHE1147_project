"""Clean one protein's Morgan + RDKit features and split into train/val/test (80/10/10).

Cleaning, in order:
  1. drop feature columns with any NaN
  2. drop rows with any NaN
  3. drop zero-variance features
The split is random and stratified on is_inhibitor (seed 0).

Reads <protein>_morgan_rdkit.csv (from featurize.py). Writes
splits/<protein>/{train,val,test}.csv.

Usage (from the repo root):
    python split.py egfr_erbB1
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
META = ["chembl_id", "smiles", "is_inhibitor"]
SEED = 0


def main(protein):
    df = pd.read_csv(ROOT / f"{protein}_morgan_rdkit.csv")
    feats = [c for c in df.columns if c not in META]

    nan_cols = [c for c in feats if df[c].isna().any()]
    df = df.drop(columns=nan_cols)
    feats = [c for c in feats if c not in nan_cols]

    n = len(df)
    df = df.dropna(subset=feats).reset_index(drop=True)
    nan_rows = n - len(df)

    zero_var = [c for c in feats if df[c].nunique() <= 1]
    df = df.drop(columns=zero_var)

    train, rest = train_test_split(df, test_size=0.2, stratify=df["is_inhibitor"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["is_inhibitor"], random_state=SEED)

    out = ROOT / "splits" / protein
    out.mkdir(parents=True, exist_ok=True)
    print(f"dropped {len(nan_cols)} NaN columns, {nan_rows} NaN rows, {len(zero_var)} zero-variance features"
          f" -> {len(df)} rows x {df.shape[1] - len(META)} features")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}: {len(part)} rows, {100 * part['is_inhibitor'].mean():.1f}% inhibitors")


if __name__ == "__main__":
    main(sys.argv[1])
