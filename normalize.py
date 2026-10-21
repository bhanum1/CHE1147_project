"""Normalize one protein's train/test splits (run after split.py's correlation filter).

All rules are decided on train and applied unchanged to val/test:
  1. Binary features (only 0 and 1 in train, e.g. Morgan bits) are left as is.
  2. For each continuous feature with train excess kurtosis > 7, try a log transform,
     sign(x) * log1p(|x|) (plain log1p for non-negative features). Keep it only if
     it brings the train kurtosis to 7 or below; otherwise leave the feature unlogged.
  3. Every continuous feature is z-scored with the train mean and std.

Reads splits/<protein>/{train,test}.csv. Writes
splits/<protein>/{train,test}_zscore.csv and zscore_params.csv (per-feature
treatment, kurtosis before/after log, train mean and std).

Usage (from the repo root):
    python normalize.py egfr_erbB1
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
META = ["chembl_id", "smiles", "is_inhibitor"]
KURTOSIS_THRESHOLD = 7


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def main(protein):
    folder = ROOT / "splits" / protein
    parts = {name: pd.read_csv(folder / f"{name}.csv") for name in ("train", "test")}
    train = parts["train"]
    feats = [c for c in train.columns if c not in META]
    binary = [c for c in feats if set(train[c].unique()) <= {0, 1}]
    continuous = [c for c in feats if c not in binary]

    kurt_before = train[continuous].kurt()
    tried = kurt_before.index[kurt_before > KURTOSIS_THRESHOLD].tolist()
    kurt_log = signed_log1p(train[tried]).kurt()
    logged = kurt_log.index[kurt_log <= KURTOSIS_THRESHOLD].tolist()
    for part in parts.values():
        part[logged] = signed_log1p(part[logged])

    scaler = StandardScaler().fit(train[continuous])
    params = pd.DataFrame({"feature": feats}).set_index("feature")
    params["treatment"] = ["binary" if c in binary else "log+zscore" if c in logged
                           else "zscore (log reverted)" if c in tried else "zscore" for c in feats]
    params.loc[continuous, "kurtosis_before"] = kurt_before
    params.loc[tried, "kurtosis_if_logged"] = kurt_log
    params.loc[continuous, "train_mean"] = scaler.mean_
    params.loc[continuous, "train_std"] = scaler.scale_
    params.reset_index().to_csv(folder / "zscore_params.csv", index=False)

    print(f"{len(binary)} binary features left as is; {len(continuous)} continuous: "
          f"{len(tried)} with kurtosis > {KURTOSIS_THRESHOLD}, {len(logged)} kept logged, "
          f"{len(tried) - len(logged)} reverted; all {len(continuous)} z-scored")
    for name, part in parts.items():
        part[continuous] = scaler.transform(part[continuous])
        part.to_csv(folder / f"{name}_zscore.csv", index=False)
        z = part[continuous].abs()
        print(f"{name}: {len(part)} rows, max |z| {z.max().max():.1f}, "
              f"{int((z > 10).any().sum())} features with |z| > 10")


if __name__ == "__main__":
    main(sys.argv[1])
