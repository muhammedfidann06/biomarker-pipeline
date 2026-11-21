"""Synthetische LC-MS-Daten zum Testen der Pipeline.

Echte Metabolomik-Daten sind meistens nicht öffentlich. Darum erzeugen wir Daten,
die ähnlich aussehen: log-normale Intensitäten, Batch-Effekte pro Probe,
Werte unter der Nachweisgrenze (= 0) und einige "echte" Biomarker.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

INTERNAL_STANDARD = "IS_d5"  # interner Standard: in jeder Probe gleich viel zugegeben


@dataclass
class Dataset:
    intensities: pd.DataFrame  # Zeilen = Proben, Spalten = Features (m/z @ RT)
    groups: pd.Series  # Gruppe pro Probe, z. B. "control" / "patient"
    true_biomarkers: list[str]  # nur bei synthetischen Daten bekannt


def make_lcms_dataset(
    n_per_group: int = 30,
    n_features: int = 300,
    n_biomarkers: int = 12,
    effect: float = 1.0,
    missing_rate: float = 0.08,
    seed: int = 0,
) -> Dataset:
    """Erzeugt einen Datensatz mit zwei Gruppen.

    effect: Unterschied der Biomarker in log2-Einheiten (1.0 = doppelt so viel).
    """
    rng = np.random.default_rng(seed)
    n = 2 * n_per_group
    groups = pd.Series(["control"] * n_per_group + ["patient"] * n_per_group, name="group")

    # Feature-Namen wie in echten Daten: Masse/Ladung und Retentionszeit
    mz = rng.uniform(80, 900, n_features)
    rt = rng.uniform(0.5, 18, n_features)
    names = [f"M{m:.4f}_RT{r:.2f}" for m, r in zip(mz, rt)]

    # Grundniveau pro Feature (log2) + biologisches Rauschen pro Probe
    base = rng.normal(16, 2, n_features)
    log2 = base + rng.normal(0, 0.6, (n, n_features))

    # Einige Features sind bei Patienten höher oder niedriger -> echte Biomarker
    marker_idx = rng.choice(n_features, n_biomarkers, replace=False)
    signs = rng.choice([-1, 1], n_biomarkers)
    log2[n_per_group:, marker_idx] += signs * effect

    # Technischer Effekt: Jede Probe hat einen anderen "Verdünnungsfaktor".
    # Genau das soll die Normalisierung später korrigieren.
    dilution = rng.normal(0, 0.5, (n, 1))
    log2 += dilution

    intensities = np.power(2.0, log2)

    # Werte unter der Nachweisgrenze werden als 0 gespeichert (wie im Gerät)
    detection_limit = np.power(2.0, base - 1.5)
    below = (intensities < detection_limit) | (rng.random((n, n_features)) < missing_rate)
    intensities[below] = 0.0

    df = pd.DataFrame(intensities, columns=names)
    df.index = [f"S{i + 1:03d}" for i in range(n)]
    groups.index = df.index

    # Interner Standard: gleiche Menge, aber auch er sieht die Verdünnung
    df[INTERNAL_STANDARD] = np.power(2.0, 20 + dilution[:, 0] + rng.normal(0, 0.05, n))

    return Dataset(df, groups, [names[i] for i in marker_idx])
