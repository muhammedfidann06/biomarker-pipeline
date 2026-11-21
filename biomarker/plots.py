"""Abbildungen: Volcano-Plot, PCA und Boxplots der besten Features."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # ohne Bildschirm (z. B. in der CI)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def volcano(result: pd.DataFrame, out: Path, q: float = 0.05, min_fc: float = 0.5) -> None:
    """x = log2 Fold Change, y = −log10(q). Rechts oben / links oben = interessant."""
    y = -np.log10(result["q_bh"].clip(lower=1e-300))
    sig = (result["q_bh"] < q) & (result["log2fc"].abs() >= min_fc)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(result["log2fc"][~sig], y[~sig], s=10, c="#9e9e9e", label="nicht signifikant")
    ax.scatter(result["log2fc"][sig], y[sig], s=14, c="#e5534b", label=f"q < {q}, |log2FC| ≥ {min_fc}")
    ax.axhline(-np.log10(q), ls="--", c="k", lw=0.8)
    for x in (-min_fc, min_fc):
        ax.axvline(x, ls="--", c="k", lw=0.8)
    ax.set_xlabel("log2 Fold Change (patient / control)")
    ax.set_ylabel("−log10(q)")
    ax.set_title("Volcano-Plot")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def pca_plot(scores: pd.DataFrame, groups: pd.Series, variance: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, colour in zip(sorted(groups.unique()), ["#1e88e5", "#e5534b", "#43a047"]):
        part = scores[groups == name]
        ax.scatter(part["PC1"], part["PC2"], label=name, c=colour, s=24, alpha=0.8)
    ax.set_xlabel(f"PC1 ({variance[0]:.0%})")
    ax.set_ylabel(f"PC2 ({variance[1]:.0%})")
    ax.set_title("PCA")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def roc_plot(y_true: np.ndarray, probability: np.ndarray, out: Path) -> float:
    """ROC-Kurve: Wie gut trennt das Modell die Gruppen bei jeder Schwelle?

    x = Falsch-Positiv-Rate, y = Richtig-Positiv-Rate. Die Diagonale ist Raten (AUC 0,5).
    Gibt die AUC zurück.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, _ = roc_curve(y_true, probability)
    auc = float(roc_auc_score(y_true, probability))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, c="#e5534b", lw=2, label=f"Random Forest (AUC = {auc:.2f})")
    ax.plot([0, 1], [0, 1], ls="--", c="#9e9e9e", label="Zufall")
    ax.set_xlabel("Falsch-Positiv-Rate (1 − Spezifität)")
    ax.set_ylabel("Richtig-Positiv-Rate (Sensitivität)")
    ax.set_title("ROC-Kurve (Kreuzvalidierung)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return auc


def top_boxplots(log_data: pd.DataFrame, groups: pd.Series, features: list[str], out: Path) -> None:
    n = len(features)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), squeeze=False)
    names = sorted(groups.unique())
    for ax, feat in zip(axes.flat, features):
        ax.boxplot([log_data.loc[groups == g, feat] for g in names], tick_labels=names)
        ax.set_title(feat, fontsize=8)
        ax.set_ylabel("log2 Intensität")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
