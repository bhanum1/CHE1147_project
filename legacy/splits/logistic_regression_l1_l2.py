#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Logistic regression with L1 and L2 regularization
=================================================
Binary classification of EGFR / erbB1 inhibitors from the pre-processed
train_zscore.csv file produced by the CHE1147 project pipeline.

Dataset facts (train_zscore.csv):
    rows      : 3 985 molecules (unique chembl_id)
    columns   : chembl_id, smiles, 2 007 Morgan bits, 66 rdkit fragment
                counts, 92 rdkit descriptors, is_inhibitor (target)
    target    : 2 756 positives (69.2 %) / 1 229 negatives (30.8 %)
    missing   : none; the continuous descriptors were already z-scored
                with the training-set statistics (see zscore_params.csv)

The script
    1. drops the two identifier columns and keeps every numeric feature,
    2. makes a stratified 80/20 hold-out split of train_zscore.csv,
    3. tunes C (inverse regularization strength) by 5-fold stratified
       cross-validation on the training part, separately for
       L1 (lasso, sparse) and L2 (ridge, dense) penalties,
    4. refits the best model of each penalty and scores the hold-out set,
    5. writes metrics, coefficients, selected features and figures.

Usage
    python logistic_regression_l1_l2.py
    python logistic_regression_l1_l2.py --cv 10 --test-size 0.25
    python logistic_regression_l1_l2.py --class-weight balanced
    python logistic_regression_l1_l2.py --test-data egfr_erbB1/test_zscore.csv
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: figures are written to disk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
TARGET = "is_inhibitor"
ID_COLUMNS = ("chembl_id", "smiles")
RANDOM_STATE = 42
PENALTIES = ("l1", "l2")
PENALTY_TITLES = {"l1": "L1 (lasso)", "l2": "L2 (ridge)"}
COLORS = {"l1": "#c0392b", "l2": "#1f6fb2"}
POS_COLOR = "#1f6fb2"   # coefficient > 0: pushes towards "inhibitor"
NEG_COLOR = "#c0392b"   # coefficient < 0: pushes towards "not inhibitor"

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "egfr_erbB1" / "train_zscore.csv"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_dataset(path: Path, target: str = TARGET) -> tuple[pd.DataFrame, pd.Series]:
    """Read a CSV split and return (X, y) with the id columns removed."""
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")

    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"target column {target!r} missing from {path}")

    drop = [c for c in ID_COLUMNS if c in df.columns]
    X = df.drop(columns=drop + [target])
    y = df[target].astype(int)

    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise ValueError(f"non-numeric features found: {non_numeric[:5]}")

    X = X.astype(np.float32)
    if X.isna().any().any():
        raise ValueError("missing values found in X")

    print(f"[data] {path.name}: {X.shape[0]} rows x {X.shape[1]} features, "
          f"{int(y.sum())} positives / {int((1 - y).sum())} negatives")
    return X, y


# --------------------------------------------------------------------------- #
# Model helpers
# --------------------------------------------------------------------------- #
def make_model(penalty: str, C: float, class_weight, seed: int = RANDOM_STATE):
    """Logistic regression with the given penalty.

    liblinear is the fast, standard L1 solver for this problem size; lbfgs is
    the standard choice for L2.  ('saga' would handle both - and elastic-net -
    but needs ~100x longer to converge on 2 000+ features.)

    intercept_scaling only affects liblinear: it appends a synthetic constant
    feature whose weight *is* the intercept, and that weight is itself
    penalized.  With the default scaling of 1, a strong L1 penalty therefore
    drives the intercept to exactly 0 (here it collapses the 69/31 prior into
    the features), which distorts the predicted probabilities.  Scaling the
    synthetic feature up by 100 makes the regularizer see it as ~100x less
    costly, so the intercept stays effectively unpenalized - the same
    treatment lbfgs gives it - at a fraction of saga's run time.
    """
    if penalty == "l1":
        return LogisticRegression(
            penalty="l1", C=C, solver="liblinear", intercept_scaling=100.0,
            class_weight=class_weight, max_iter=5000, tol=1e-4, random_state=seed,
        )
    return LogisticRegression(
        penalty="l2", C=C, solver="lbfgs",
        class_weight=class_weight, max_iter=5000, tol=1e-4, random_state=seed,
    )


def metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Threshold-independent and threshold-dependent metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob),
        "log_loss": log_loss(y_true, np.clip(y_prob, 1e-15, 1 - 1e-15)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f1_neg": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def nonzero(coef: np.ndarray, tol: float = 1e-8) -> int:
    """Number of features with a non-zero coefficient (L1 sparsity)."""
    return int(np.sum(np.abs(coef) > tol))


def coefficient_table(model, features, penalty: str, n_features: int) -> pd.DataFrame:
    """Selected features sorted by |coefficient|, largest first."""
    coef = pd.Series(model.coef_.ravel(), index=features, name="coef")
    table = (
        coef[coef.abs() > 1e-8]
        .sort_values(key=np.abs, ascending=False)
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    table["penalty"] = penalty
    table["n_features"] = n_features
    table["intercept"] = float(model.intercept_[0])
    return table


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_cv_curves(cv_tables: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for p, table in cv_tables.items():
        ax.errorbar(
            table["C"], table["mean_test_score"], yerr=table["std_test_score"],
            marker="o", ms=4, capsize=3, lw=1.6,
            color=COLORS[p], label=f"{PENALTY_TITLES[p]}",
        )
    ax.set_xscale("log")
    ax.set_xlabel("C  (inverse regularization strength: smaller C = stronger penalty)")
    ax.set_ylabel("5-fold CV ROC AUC")
    ax.set_title("Hyper-parameter tuning of C by stratified cross-validation")
    ax.grid(alpha=0.3, ls=":")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "cv_auc_vs_C.png", dpi=200)
    plt.close(fig)


def plot_regularization_path(paths: dict, out: Path) -> None:
    """How many features survive as the penalty gets weaker."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (p, table) in zip(axes, paths.items()):
        ax.plot(table["C"], table["n_nonzero"], marker="o", ms=4,
                color=COLORS[p], lw=1.8)
        ax.set_xscale("log")
        ax.set_xlabel("C")
        ax.set_ylabel("non-zero coefficients")
        ax.set_title(f"{PENALTY_TITLES[p]}: sparsity path")
        ax.grid(alpha=0.3, ls=":")
        ax.axhline(table["n_features"].iloc[0], color="grey", ls="--", lw=1,
                   label="all features")
        ax.set_ylim(0, table["n_features"].iloc[0] * 1.08)  # common scale
        ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "regularization_path.png", dpi=200)
    plt.close(fig)


def plot_roc_pr(y_test, probs: dict, scores: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    ax = axes[0]
    for p, prob in probs.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        ax.plot(fpr, tpr, lw=2, color=COLORS[p],
                label=f"{PENALTY_TITLES[p]}  (AUC = {scores[p]['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves - hold-out set")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.3, ls=":")

    ax = axes[1]
    for p, prob in probs.items():
        prec, rec, _ = precision_recall_curve(y_test, prob)
        ax.plot(rec, prec, lw=2, color=COLORS[p],
                label=f"{PENALTY_TITLES[p]}  (AP = {scores[p]['average_precision']:.4f})")
    base = float(np.mean(y_test))
    ax.axhline(base, color="k", ls="--", lw=1, alpha=0.6,
               label=f"prevalence = {base:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves - hold-out set")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(alpha=0.3, ls=":")

    fig.tight_layout()
    fig.savefig(out / "roc_pr_curves.png", dpi=200)
    plt.close(fig)


def plot_confusion(y_test, probs: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, len(probs), figsize=(5 * len(probs), 4.3))
    axes = np.atleast_1d(axes)
    for ax, (p, prob) in zip(axes, probs.items()):
        cm = confusion_matrix(y_test, (prob >= 0.5).astype(int), labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=13,
                    color="white" if v > cm.max() * 0.55 else "black")
        ax.set_xticks([0, 1], ["0", "1"])
        ax.set_yticks([0, 1], ["0", "1"])
        ax.set_xlabel("Predicted label (threshold 0.5)")
        ax.set_ylabel("True label")
        ax.set_title(f"{PENALTY_TITLES[p]} confusion matrix")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=200)
    plt.close(fig)


def plot_coefficients(coef_tables: dict, top_n: int, out: Path) -> None:
    """Side-by-side top-|coef| features; shows how much sparser L1 is."""
    fig, axes = plt.subplots(1, len(coef_tables), figsize=(13, 6.4))
    axes = np.atleast_1d(axes)
    for ax, (p, table) in zip(axes, coef_tables.items()):
        top = table.head(top_n)

        # Labels are ugly rdkit names; add the feature family as a prefix.
        def family(name: str) -> str:
            if name.startswith("morgan_"):
                return "FP"
            if name.startswith("fr_"):
                return "FR"
            return "DESC"

        labels = [f"{family(r.feature)}:{r.feature}" for r in top.itertuples()]
        y = np.arange(len(top))

        # Descending |coef| => larger on top.
        ax.barh(y, top["coef"], color=np.where(top["coef"] > 0, POS_COLOR, NEG_COLOR))
        ax.set_yticks(y, labels, fontsize=7.5)
        ax.invert_yaxis()
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("coefficient (log-odds per z-score / per bit)")
        ax.set_title(f"{PENALTY_TITLES[p]}: top {top_n} |coef|\n"
                     f"({int(table.shape[0]):,} selected of {int(table['n_features'].iloc[0]):,})")
        ax.grid(alpha=0.25, ls=":", axis="x")
        ax.legend(handles=[Patch(color=POS_COLOR, label="positive: favours inhibitor"),
                           Patch(color=NEG_COLOR, label="negative: disfavours inhibitor")],
                  frameon=False, fontsize=7.5, loc="best")
    fig.tight_layout()
    fig.savefig(out / "top_coefficients.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="L1 vs L2 regularized logistic regression on train_zscore.csv")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA,
                    help=f"training csv (default: {DEFAULT_DATA})")
    ap.add_argument("--test-data", type=Path, default=None,
                    help="optional csv to evaluate on (e.g. egfr_erbB1/test_zscore.csv); "
                         "when given, the model is refit on ALL of --data")
    ap.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--test-size", type=float, default=0.20,
                    help="hold-out fraction (default 0.20)")
    ap.add_argument("--cv", type=int, default=5, help="CV folds (default 5)")
    ap.add_argument("--class-weight", choices=("none", "balanced"), default="none",
                    help="'balanced' up-weights the minority class")
    ap.add_argument("--top-n", type=int, default=20,
                    help="features shown per model in the coefficient plot")
    ap.add_argument("--jobs", type=int, default=-1,
                    help="parallel jobs for the grid search (-1 = all cores, 1 = serial)")
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    ap.add_argument("--no-plots", action="store_true", help="skip figure generation")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    class_weight = None if args.class_weight == "none" else "balanced"

    # ------------------------------------------------------------------ data
    X, y = load_dataset(args.data)
    n_features = X.shape[1]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed)
    print(f"[split] train {X_tr.shape[0]} / test {X_te.shape[0]} "
          f"(test prevalence {y_te.mean():.3f})")

    C_grid = np.logspace(-3, 2, 10)
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)

    # Grid search: one pass per penalty.  AUC is used because it is
    # threshold-independent and insensitive to the 69/31 class imbalance.
    best_models, cv_rows, cv_tables, path_tables, coef_tables = {}, [], {}, {}, {}

    for penalty in PENALTIES:
        print(f"\n[cv] {PENALTY_TITLES[penalty]}: searching {len(C_grid)} values of C "
              f"over {args.cv} folds ...")
        search = GridSearchCV(
            make_model(penalty, 1.0, class_weight, args.seed),
            param_grid={"C": C_grid},
            scoring="roc_auc",
            cv=cv,
            n_jobs=args.jobs,
            refit=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            search.fit(X_tr, y_tr)

        table = pd.DataFrame(search.cv_results_)[
            ["param_C", "mean_test_score", "std_test_score", "rank_test_score"]
        ].rename(columns={"param_C": "C"})
        table["penalty"] = penalty
        cv_tables[penalty] = table
        cv_rows.append(table)

        best_C = float(search.best_params_["C"])
        model = search.best_estimator_
        best_models[penalty] = model
        std_at_best = float(table.loc[np.isclose(table["C"], best_C),
                                      "std_test_score"].iloc[0])
        print(f"[cv] {penalty}: best C = {best_C:g}, "
              f"CV AUC = {search.best_score_:.4f} +/- {std_at_best:.4f}")

        # Sparsity path: refit each C on the full training split.
        rows = []
        for C in C_grid:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                m = make_model(penalty, float(C), class_weight, args.seed).fit(X_tr, y_tr)
            rows.append({"C": float(C), "n_nonzero": nonzero(m.coef_.ravel()),
                         "n_features": n_features})
        path_tables[penalty] = pd.DataFrame(rows)

        coef_table = coefficient_table(model, X.columns, penalty, n_features)
        coef_tables[penalty] = coef_table
        coef_table.drop(columns=["n_features"]).to_csv(
            out / f"coefficients_{penalty}.csv", index=False)
        print(f"[sparse] {penalty}: {len(coef_table):,} of {n_features:,} features kept "
              f"({100 * len(coef_table) / n_features:.1f}%), intercept "
              f"{float(model.intercept_[0]):+.4f}")

    # ------------------------------------------------------------- evaluation
    eval_sets = {}
    if args.test_data is None:
        eval_sets["holdout"] = (X_te, y_te)
        fitted = best_models
    else:
        # User supplied an extra split -> refit on everything available.
        X_extra, y_extra = load_dataset(args.test_data)
        eval_sets["test_data"] = (X_extra, y_extra)
        fitted = {}
        for penalty, model in best_models.items():
            refit = make_model(penalty, model.C, class_weight, args.seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                refit.fit(X, y)
            fitted[penalty] = refit
        print(f"[eval] refit both models on all {X.shape[0]} training rows")

        # Keep the exported coefficients consistent with the models being
        # scored, otherwise they would describe the 80/20-split fits instead.
        for penalty in PENALTIES:
            coef_tables[penalty] = coefficient_table(
                fitted[penalty], X.columns, penalty, n_features)
            coef_tables[penalty].drop(columns=["n_features"]).to_csv(
                out / f"coefficients_{penalty}.csv", index=False)
            print(f"[sparse] {penalty} (refit): {len(coef_tables[penalty]):,} of "
                  f"{n_features:,} features kept, intercept "
                  f"{float(fitted[penalty].intercept_[0]):+.4f}")

    summary_rows, probs_all, scores_all, eval_name = [], {}, {}, ""
    for set_name, (Xs, ys) in eval_sets.items():
        probs = {p: fitted[p].predict_proba(Xs)[:, 1] for p in PENALTIES}
        scores = {p: metrics(ys, probs[p]) for p in PENALTIES}
        probs_all, scores_all, eval_name = probs, scores, set_name

        for p in PENALTIES:
            row = {"evaluation_set": set_name, "penalty": p,
                   "best_C": float(fitted[p].C),
                   "n_selected": int(len(coef_tables[p])),
                   "n_features": int(n_features)}
            row.update(scores[p])
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "metrics_summary.csv", index=False)
    pd.concat(cv_rows).to_csv(out / "cv_results.csv", index=False)
    for p, t in path_tables.items():
        t.to_csv(out / f"sparsity_path_{p}.csv", index=False)

    # ---------------------------------------------------------------- report
    cols = ["penalty", "best_C", "n_selected", "roc_auc", "average_precision",
            "accuracy", "precision", "recall", "f1", "log_loss"]
    print("\n" + "=" * 78)
    print(f"PERFORMANCE ON '{eval_name}' (threshold = 0.5)")
    print("=" * 78)
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    for p in PENALTIES:
        print(f"\n--- {PENALTY_TITLES[p]} ---")
        print(f"confusion matrix [tn fp; fn tp]: {scores_all[p]['tn']} "
              f"{scores_all[p]['fp']} {scores_all[p]['fn']} {scores_all[p]['tp']}")
        print("top 8 features by |coef|:")
        print(coef_tables[p].head(8)[["feature", "coef"]]
              .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    # Which features does L1 keep that L2 also rates highly?
    l1_top = set(coef_tables["l1"].head(args.top_n)["feature"])
    l2_ranked = coef_tables["l2"]["feature"].tolist()
    overlap = len(l1_top.intersection(set(l2_ranked[:args.top_n])))
    print(f"\n[compare] top-{args.top_n} L1 features also in top-{args.top_n} "
          f"L2 features: {overlap}/{args.top_n}")

    # --------------------------------------------------------------- figures
    if not args.no_plots:
        plot_cv_curves(cv_tables, out)
        plot_regularization_path(path_tables, out)
        plot_roc_pr(list(eval_sets.values())[0][1], probs_all, scores_all, out)
        plot_confusion(list(eval_sets.values())[0][1], probs_all, out)
        plot_coefficients(coef_tables, args.top_n, out)
        print(f"\n[figures] written to {out}")

    print(f"\n[done] artifacts in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
