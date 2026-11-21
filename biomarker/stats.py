"""Statistische Tests pro Feature und Korrektur für multiples Testen."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def bonferroni(p: np.ndarray) -> np.ndarray:
    """Sehr streng: jeder p-Wert mal Anzahl Tests (maximal 1)."""
    p = np.asarray(p, dtype=float)
    return np.minimum(p * len(p), 1.0)


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """False Discovery Rate nach Benjamini-Hochberg.

    Schritte:
    1. p-Werte sortieren
    2. q_i = p_i * m / Rang_i
    3. Von hinten nach vorne das Minimum nehmen (damit q monoton bleibt)
    """
    p = np.asarray(p, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.minimum(ranked, 1.0)
    return q


def normality(log_data: pd.DataFrame, groups: pd.Series, alpha: float = 0.05) -> pd.Series:
    """Anteil der Features, die in ALLEN Gruppen normalverteilt sind (Shapiro-Wilk).

    Ist der Anteil klein, nehmen wir lieber einen nicht-parametrischen Test.
    """
    normal = pd.Series(True, index=log_data.columns)
    for _, part in log_data.groupby(groups):
        p = part.apply(lambda col: stats.shapiro(col).pvalue)
        normal &= p > alpha
    return normal


def differential_analysis(
    log_data: pd.DataFrame,
    groups: pd.Series,
    case: str,
    control: str,
    test: str = "auto",
) -> pd.DataFrame:
    """Vergleicht jedes Feature zwischen zwei Gruppen.

    test: "welch" (t-Test), "mannwhitney" oder "auto" (Welch, wenn > 80 % normalverteilt).
    Ergebnis-Tabelle sortiert nach q-Wert.
    """
    a = log_data[groups == case]
    b = log_data[groups == control]
    if test == "auto":
        share_normal = normality(log_data[groups.isin([case, control])], groups).mean()
        test = "welch" if share_normal > 0.8 else "mannwhitney"

    if test == "welch":
        p = stats.ttest_ind(a, b, equal_var=False).pvalue
    elif test == "mannwhitney":
        p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
    else:
        raise ValueError(f"Unbekannter Test: {test}")

    # Daten sind log2 -> Differenz der Mittelwerte = log2 Fold Change
    log2fc = a.mean() - b.mean()
    result = pd.DataFrame(
        {
            "log2fc": log2fc,
            "p": p,
            "q_bh": benjamini_hochberg(p),
            "p_bonferroni": bonferroni(p),
        },
        index=log_data.columns,
    ).sort_values("q_bh")
    result.attrs["test"] = test  # merken, welcher Test benutzt wurde
    return result


def kruskal_wallis(log_data: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Für drei oder mehr Gruppen: Kruskal-Wallis (nicht-parametrische ANOVA)."""
    parts = [part for _, part in log_data.groupby(groups)]
    p = np.array([stats.kruskal(*[g[col] for g in parts]).pvalue for col in log_data.columns])
    return pd.DataFrame({"p": p, "q_bh": benjamini_hochberg(p)}, index=log_data.columns).sort_values("q_bh")


def significant(result: pd.DataFrame, q: float = 0.05, min_abs_log2fc: float = 0.5) -> pd.DataFrame:
    """Signifikant UND groß genug (sonst ist der Effekt biologisch egal)."""
    return result[(result["q_bh"] < q) & (result["log2fc"].abs() >= min_abs_log2fc)]
