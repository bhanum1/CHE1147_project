"""Figure A1: key descriptors after normalization, by class, EGFR training split.

Shows what the preprocessing actually does to the continuous descriptors, and
that the two classes overlap heavily on all five of them -- which is the point
of the figure: no single descriptor separates inhibitors from non-inhibitors, so
the signal the models find has to be multivariate.

The z-scores are computed here rather than read from a pre-scaled table, over
the random split's 3985 training molecules and with the population standard
deviation (`ddof=0`), which is what `MolecularPreprocessor` uses. Only the
training molecules contribute the mean and scale, so the figure shows the same
statistics a model would have been fitted with.

Usage (from the repo root):
    python report/appendix/make_figA.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/egfr_erbB1"
OUT = Path(__file__).parent / "figA1_descriptor_distributions.png"

# RDKit descriptor name -> the short label used in the paper.
KEY = {"MolWt": "MW", "MolLogP": "LogP", "TPSA": "TPSA",
       "NumHAcceptors": "HBA", "NumRotatableBonds": "Rot. bonds"}
CLASS = [(0, "#4c72b0", "Non-inhibitor"), (1, "#c44e52", "Inhibitor")]

features = pd.read_csv(DATA / "features.csv", usecols=["chembl_id", "is_inhibitor", *KEY])
manifest = pd.read_csv(DATA / "split_manifest.csv")
train = features[features.chembl_id.isin(manifest.loc[manifest.split == "train", "chembl_id"])]

raw = train[list(KEY)]
z = (raw - raw.mean()) / raw.std(ddof=0)
y = train.is_inhibitor.to_numpy()

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(7, 4.2))

# Two boxes per descriptor, nudged left and right of the tick so the classes sit
# side by side rather than on top of each other.
pos = np.arange(len(KEY))
for k, (lab, col, name) in enumerate(CLASS):
    ax.boxplot([z.loc[y == lab, c] for c in KEY], positions=pos + (k - 0.5) * 0.36, widths=0.3,
               patch_artist=True,
               boxprops={"facecolor": col, "alpha": 0.75, "lw": 0.6},
               medianprops={"color": "k", "lw": 1},
               whiskerprops={"lw": 0.6}, capprops={"lw": 0.6},
               flierprops={"marker": ".", "ms": 2.5, "mfc": col, "mec": "none", "alpha": 0.5})
    # Proxy artist: boxplot draws no legend handle of its own.
    ax.plot([], [], "s", color=col, alpha=0.75, label=name)

# Shade the +/- 3 sigma band, so outliers are visible as outliers.
ax.axhspan(-3, 3, color="0.93", zorder=0)
ax.set_xticks(pos, KEY.values())
ax.set_ylabel("z-score")
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
ax.text(pos[-1] + 0.45, -2.85, "|z| ≤ 3", ha="right", va="bottom", fontsize=8, color="0.45")

fig.tight_layout()
fig.savefig(OUT, dpi=200)
print(f"wrote {OUT}")
