"""EDA for one protein's Morgan + RDKit feature file, following the course pipeline
(CHE1147 tutorial_03_eda_and_rdkit, sections 5-7).

Steps: SMILES canonicalization and duplicates, dataset overview and missing values,
class balance, key descriptor distributions, correlations, Lipinski Rule of Five,
LogP vs TPSA by class, fingerprint bits, Tanimoto similarity, ydata-profiling report.

Reads the unfiltered processed feature table. Writes plots, summary.json and
an optional ydata-profiling report to reports/eda/<protein>/.

Usage (from the repo root):
    python -m egfr_pipeline.eda egfr_erbB1
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem, DataStructs, RDLogger
from sklearn.metrics import roc_auc_score

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
# Tutorial's key descriptors -> RDKit column names in the feature file
KEY = {"MW": "MolWt", "LogP": "MolLogP", "TPSA": "TPSA", "HBD": "NumHDonors",
       "HBA": "NumHAcceptors", "RotBonds": "NumRotatableBonds"}
LABELS = {1: "Inhibitor", 0: "Non-inhibitor"}
COLORS = {1: "#d62728", 0: "#1f77b4"}


def canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else None


def tanimoto_stats(bits, y):
    """Nearest-neighbour and within/between-class Tanimoto similarity."""
    fps = []
    for row in bits:
        fp = DataStructs.ExplicitBitVect(len(row))
        fp.SetBitsFromList(np.flatnonzero(row).tolist())
        fps.append(fp)

    n = len(fps)
    nn_sim = np.zeros(n)
    nn_same = np.zeros(n, dtype=bool)
    same_sum = diff_sum = 0.0
    same_n = diff_n = 0
    for i in range(n):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps))
        sims[i] = -1
        j = sims.argmax()
        nn_sim[i], nn_same[i] = sims[j], y[j] == y[i]
        upper = sims[i + 1:]
        same = y[i + 1:] == y[i]
        same_sum += upper[same].sum(); same_n += same.sum()
        diff_sum += upper[~same].sum(); diff_n += (~same).sum()
    return nn_sim, nn_same, same_sum / same_n, diff_sum / diff_n


def main(protein):
    out = ROOT / "reports" / "eda" / protein
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data" / "processed" / protein / "features.csv")
    fp_cols = [c for c in df.columns if c.startswith("morgan_")]
    desc_cols = [c for c in df.columns if c not in fp_cols + ["chembl_id", "smiles", "is_inhibitor"]]
    y = df["is_inhibitor"].to_numpy()
    s = {"protein": protein, "n_molecules": len(df), "n_fp_bits": len(fp_cols),
         "n_descriptors": len(desc_cols)}

    # 1. Canonicalization and duplicates
    df["canonical_smiles"] = df["smiles"].apply(canonical)
    dup = df[df.duplicated("canonical_smiles", keep=False)]
    conflict = dup.groupby("canonical_smiles")["is_inhibitor"].nunique().gt(1).sum()
    s["canonicalization"] = {
        "smiles_changed": int((df["smiles"] != df["canonical_smiles"]).sum()),
        "unique_smiles": int(df["smiles"].nunique()),
        "unique_canonical": int(df["canonical_smiles"].nunique()),
        "duplicate_structures": int(dup["canonical_smiles"].nunique()),
        "duplicate_rows": len(dup),
        "duplicates_with_conflicting_labels": int(conflict),
        "examples": dup.sort_values("canonical_smiles")[["chembl_id", "canonical_smiles", "is_inhibitor"]]
                       .head(6).to_dict("records"),
    }

    # 2. Overview and quality
    desc = df[desc_cols]
    missing = desc.isna().sum()
    constant = [c for c in desc_cols if desc[c].nunique(dropna=True) <= 1]
    corr_all = desc.drop(columns=constant).corr().abs()
    upper = corr_all.where(np.triu(np.ones(corr_all.shape, dtype=bool), 1))
    s["quality"] = {
        "descriptor_cols_with_missing": {c: int(v) for c, v in missing[missing > 0].items()},
        "rows_with_any_missing": int(desc.isna().any(axis=1).sum()),
        "constant_descriptors": constant,
        "descriptor_pairs_abs_r_gt_0.95": int((upper > 0.95).sum().sum()),
    }

    # 3. Class balance
    counts = df["is_inhibitor"].value_counts()
    s["class_balance"] = {LABELS[k]: {"n": int(v), "pct": round(100 * v / len(df), 1)}
                          for k, v in counts.items()}

    # 4. Key descriptor stats (overall + per class) and univariate separability
    key = df[list(KEY.values())].rename(columns={v: k for k, v in KEY.items()})
    key["is_inhibitor"] = y
    s["key_descriptors"] = key.drop(columns="is_inhibitor").describe().round(2).to_dict()
    s["key_by_class_median"] = key.groupby("is_inhibitor").median().rename(index=LABELS).round(2).to_dict("index")

    aucs = {}
    for c in desc_cols:
        x = desc[c]
        m = x.notna()
        if x[m].nunique() > 1:
            a = roc_auc_score(y[m], x[m])
            aucs[c] = max(a, 1 - a)
    s["top_descriptors_by_univariate_auc"] = {k: round(v, 3) for k, v in
                                              sorted(aucs.items(), key=lambda kv: -kv[1])[:10]}
    s["key_descriptor_univariate_auc"] = {k: round(aucs[v], 3) for k, v in KEY.items()}

    # Histograms, tutorial style, with class overlay
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"{protein}: distribution of key descriptors", fontsize=16, fontweight="bold")
    for ax, name in zip(axes.flat, KEY):
        bins = np.histogram_bin_edges(key[name].dropna(), bins=30)
        for lab in (0, 1):
            ax.hist(key.loc[key.is_inhibitor == lab, name], bins=bins, alpha=0.55, density=True,
                    color=COLORS[lab], edgecolor="black", linewidth=0.3, label=LABELS[lab])
        ax.axvline(key[name].mean(), color="k", ls="--", lw=1, label=f"Mean {key[name].mean():.1f}")
        ax.axvline(key[name].median(), color="orange", ls="--", lw=1, label=f"Median {key[name].median():.1f}")
        ax.set_title(name); ax.set_ylabel("Density"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "descriptor_histograms.png", dpi=120); plt.close(fig)

    # 5. Correlation heatmap of key descriptors
    cm = key.drop(columns="is_inhibitor").corr()
    s["key_correlations_abs_gt_0.7"] = {f"{a}~{b}": round(cm.loc[a, b], 2) for i, a in enumerate(cm)
                                        for b in cm.columns[i + 1:] if abs(cm.loc[a, b]) > 0.7}
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, mask=np.triu(np.ones_like(cm, dtype=bool)), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, square=True, cbar_kws={"label": "Pearson r"})
    plt.title(f"{protein}: key descriptor correlations", fontweight="bold")
    plt.tight_layout(); plt.savefig(out / "correlation_heatmap.png", dpi=120); plt.close()

    # 6. Lipinski Rule of Five
    lip = (key.MW <= 500) & (key.LogP <= 5) & (key.HBD <= 5) & (key.HBA <= 10)
    s["lipinski_pass_pct"] = {"all": round(100 * lip.mean(), 1),
                              **{LABELS[k]: round(100 * lip[y == k].mean(), 1) for k in (1, 0)}}

    # 7. LogP vs TPSA by class (Mini Exercise 3)
    plt.figure(figsize=(8, 6))
    for lab in (0, 1):
        sub = key[key.is_inhibitor == lab]
        plt.scatter(sub.LogP, sub.TPSA, s=6, alpha=0.4, c=COLORS[lab], label=LABELS[lab])
    plt.xlabel("LogP"); plt.ylabel("TPSA (Å²)"); plt.legend(); plt.grid(alpha=0.3)
    plt.title(f"{protein}: LogP vs TPSA", fontweight="bold")
    plt.tight_layout(); plt.savefig(out / "logp_vs_tpsa.png", dpi=120); plt.close()

    # 8. Fingerprint bits
    bits = df[fp_cols].to_numpy(dtype=np.uint8)
    on = bits.sum(axis=1)
    freq = bits.mean(axis=0)
    s["fingerprint"] = {
        "bits_on_per_molecule_mean": round(float(on.mean()), 1),
        "bits_on_per_molecule_range": [int(on.min()), int(on.max())],
        "density_pct": round(100 * float(bits.mean()), 2),
        "bits_never_set": int((freq == 0).sum()),
        "bits_set_in_lt_1pct": int((freq < 0.01).sum()),
        "bits_set_in_gt_50pct": int((freq > 0.5).sum()),
    }
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(bits[:50, :200], cmap="Blues", aspect="auto")
    ax.set_xlabel("Bit index"); ax.set_ylabel("Molecule index")
    ax.set_title(f"{protein}: Morgan fingerprints (50 molecules × 200 bits)")
    fig.tight_layout(); fig.savefig(out / "fingerprint_heatmap.png", dpi=120); plt.close(fig)

    # 9. Tanimoto similarity
    nn_sim, nn_same, same_mean, diff_mean = tanimoto_stats(bits, y)
    s["tanimoto"] = {
        "nearest_neighbour_mean": round(float(nn_sim.mean()), 3),
        "nearest_neighbour_median": round(float(np.median(nn_sim)), 3),
        "pct_with_nn_ge_0.7": round(100 * float((nn_sim >= 0.7).mean()), 1),
        "pct_with_nn_eq_1.0": round(100 * float((nn_sim >= 0.999).mean()), 1),
        "nn_same_label_pct": round(100 * float(nn_same.mean()), 1),
        "mean_pairwise_same_class": round(float(same_mean), 3),
        "mean_pairwise_diff_class": round(float(diff_mean), 3),
    }
    plt.figure(figsize=(8, 5))
    plt.hist(nn_sim, bins=50, color="#4c72b0", edgecolor="black", linewidth=0.3)
    plt.xlabel("Tanimoto similarity to nearest neighbour"); plt.ylabel("Molecules")
    plt.title(f"{protein}: nearest-neighbour similarity", fontweight="bold"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out / "nn_similarity.png", dpi=120); plt.close()

    # 10. ydata-profiling on the key descriptors + label (as in the tutorial)
    try:
        from ydata_profiling import ProfileReport
    except ImportError:
        print('Optional ydata-profiling is unavailable; core EDA outputs are still generated.')
    else:
        ProfileReport(key, title=f"{protein} EDA Report", explorative=True, progress_bar=False).to_file(out / 'eda_report.html')

    (out / "summary.json").write_text(json.dumps(s, indent=2, default=str))
    print(json.dumps(s, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1])
