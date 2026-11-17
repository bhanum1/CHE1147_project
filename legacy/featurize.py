"""Morgan fingerprint + RDKit 2D descriptor features for one protein.

Matches the course baseline (projects/baseline_xgb.py): 2048-bit Morgan fingerprint
(radius 2) plus every descriptor in Descriptors.CalcMolDescriptors.

SMILES come from <protein>_features.csv; labels come from data/<protein>.csv.
Writes <protein>_morgan_rdkit.csv.

Usage (from the repo root):
    python featurize.py egfr_erbB1
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).parent
RADIUS, N_BITS = 2, 2048

morgan = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)


def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = morgan.GetFingerprintAsNumPy(mol).astype(np.uint8)
    desc = Descriptors.CalcMolDescriptors(mol)
    return {**{f"morgan_{i}": b for i, b in enumerate(fp)}, **desc}


def main(protein):
    smiles = pd.read_csv(ROOT / f"{protein}_features.csv", usecols=["chembl_id", "smiles"])
    labels = pd.read_csv(ROOT / "data" / f"{protein}.csv")
    df = labels.merge(smiles, on="chembl_id")

    rows = [featurize(s) for s in df["smiles"]]
    ok = [r is not None for r in rows]
    feats = pd.DataFrame([r for r in rows if r is not None])
    feats = feats.replace([np.inf, -np.inf], np.nan)

    out = pd.concat([df.loc[ok, ["chembl_id", "smiles"]].reset_index(drop=True), feats,
                     df.loc[ok, ["is_inhibitor"]].reset_index(drop=True)], axis=1)
    path = ROOT / f"{protein}_morgan_rdkit.csv"
    out.to_csv(path, index=False)
    print(f"{len(out)}/{len(labels)} molecules featurized "
          f"({len(labels) - len(df)} without SMILES, {ok.count(False)} unparseable) -> {path.name}")


if __name__ == "__main__":
    main(sys.argv[1])
