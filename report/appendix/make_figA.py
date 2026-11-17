"""Figure A1: key descriptors after normalization, by class, EGFR training split.

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
OUT = Path(__file__).parent / "figA1_descriptor_distributions.png"
KEY = {"MolWt": "MW", "MolLogP": "LogP", "TPSA": "TPSA", "NumHAcceptors": "HBA", "NumRotatableBonds": "Rot. bonds"}
CLASS = [(0, "#4c72b0", "Non-inhibitor"), (1, "#c44e52", "Inhibitor")]

z = pd.read_csv(ROOT / "legacy/splits/egfr_erbB1/train_zscore.csv")
y = z.is_inhibitor

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(7, 4.2))

pos = np.arange(len(KEY))
for k, (lab, col, name) in enumerate(CLASS):
    ax.boxplot([z.loc[y == lab, c] for c in KEY], positions=pos + (k - 0.5) * 0.36, widths=0.3, patch_artist=True,
               boxprops={"facecolor": col, "alpha": 0.75, "lw": 0.6}, medianprops={"color": "k", "lw": 1},
               whiskerprops={"lw": 0.6}, capprops={"lw": 0.6},
               flierprops={"marker": ".", "ms": 2.5, "mfc": col, "mec": "none", "alpha": 0.5})
    ax.plot([], [], "s", color=col, alpha=0.75, label=name)
ax.axhspan(-3, 3, color="0.93", zorder=0)
ax.set_xticks(pos, KEY.values()); ax.set_ylabel("z-score")
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
ax.text(pos[-1] + 0.45, -2.85, "|z| ≤ 3", ha="right", va="bottom", fontsize=8, color="0.45")

fig.tight_layout()
fig.savefig(OUT, dpi=200)
