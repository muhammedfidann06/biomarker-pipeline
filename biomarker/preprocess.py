"""Vorverarbeitung: Filtern, Imputation, Normalisierung, Log-Transformation.

Die Reihenfolge ist wichtig:
1. Features mit zu vielen Nullen entfernen (Zero-Thresholding)
2. Fehlende Werte auffüllen (Imputation)
3. Normalisieren (interner Standard oder Median)
4. log2 – danach sind die Daten ungefähr normalverteilt
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def zero_threshold(df: pd.DataFrame, groups: pd.Series, max_zero_fraction: float = 0.3) -> pd.DataFrame:
    """Behalte ein Feature, wenn es in MINDESTENS EINER Gruppe genug Werte hat.

    Warum pro Gruppe? Ein Biomarker kann in einer Gruppe fast immer fehlen –
    genau das ist dann interessant und darf nicht weggefiltert werden.
    """
    zero_frac = (df == 0).groupby(groups).mean()  # Anteil Nullen pro Gruppe und Feature
    keep = (zero_frac <= max_zero_fraction).any(axis=0)
    return df.loc[:, keep]


def impute_half_minimum(df: pd.DataFrame) -> pd.DataFrame:
    """Nullen durch die Hälfte des kleinsten gemessenen Wertes ersetzen.

    Idee: Der Wert war zu klein für das Gerät, also irgendwo zwischen 0 und dem Minimum.
    Einfach und in der Metabolomik sehr üblich.
    """
    out = df.replace(0, np.nan)
    half_min = out.min(axis=0) / 2
    return out.fillna(half_min)


def impute_knn(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Alternative: k-nächste-Nachbarn-Imputation (ähnliche Proben helfen)."""
    from sklearn.impute import KNNImputer

    values = KNNImputer(n_neighbors=k).fit_transform(np.log2(df.replace(0, np.nan)))
    return pd.DataFrame(np.power(2.0, values), index=df.index, columns=df.columns)


def normalize_internal_standard(df: pd.DataFrame, standard: str) -> pd.DataFrame:
    """Jede Probe durch ihren internen Standard teilen.

    Der Standard wurde in gleicher Menge zugegeben. Ist er in einer Probe größer,
    war die ganze Probe "stärker" – also teilen wir alles durch ihn.
    """
    ratio = df.drop(columns=[standard]).div(df[standard], axis=0)
    return ratio * df[standard].median()  # zurück auf die ursprüngliche Skala


def normalize_median(df: pd.DataFrame) -> pd.DataFrame:
    """Jede Probe so skalieren, dass alle Proben den gleichen Median haben."""
    medians = df.median(axis=1)
    return df.div(medians, axis=0) * medians.median()


def log2_transform(df: pd.DataFrame) -> pd.DataFrame:
    return np.log2(df)


def preprocess(
    df: pd.DataFrame,
    groups: pd.Series,
    standard: str | None = None,
    imputation: str = "half_min",
    max_zero_fraction: float = 0.3,
) -> pd.DataFrame:
    """Die ganze Vorverarbeitung in einem Schritt. Ergebnis: log2-Werte ohne Lücken."""
    features = df.drop(columns=[standard]) if standard else df
    filtered = zero_threshold(features, groups, max_zero_fraction)
    imputed = impute_knn(filtered) if imputation == "knn" else impute_half_minimum(filtered)
    if standard:
        imputed[standard] = df[standard]
        normalized = normalize_internal_standard(imputed, standard)
    else:
        normalized = normalize_median(imputed)
    return log2_transform(normalized)
