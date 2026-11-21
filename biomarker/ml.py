"""Maschinelles Lernen: PCA und Random Forest mit Kreuzvalidierung."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def pca_scores(log_data: pd.DataFrame, n_components: int = 2) -> tuple[pd.DataFrame, np.ndarray]:
    """PCA auf standardisierten Daten. Gibt Scores und erklärte Varianz zurück."""
    scaled = StandardScaler().fit_transform(log_data)
    pca = PCA(n_components=n_components, random_state=0)
    scores = pca.fit_transform(scaled)
    cols = [f"PC{i + 1}" for i in range(n_components)]
    return pd.DataFrame(scores, index=log_data.index, columns=cols), pca.explained_variance_ratio_


@dataclass
class ForestResult:
    auc_per_fold: list[float]
    importance: pd.Series  # mittlere Wichtigkeit über alle Folds, sortiert

    @property
    def mean_auc(self) -> float:
        return float(np.mean(self.auc_per_fold))


def random_forest_cv(
    log_data: pd.DataFrame,
    groups: pd.Series,
    positive: str,
    folds: int = 5,
    seed: int = 0,
) -> ForestResult:
    """Random Forest mit stratifizierter Kreuzvalidierung.

    Wichtig: Die AUC wird nur auf den Test-Folds berechnet. Sonst wäre das Ergebnis
    zu optimistisch (der Wald hat die Trainingsdaten "auswendig gelernt").
    """
    X = log_data.to_numpy()
    y = (groups == positive).to_numpy().astype(int)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    aucs: list[float] = []
    importances = np.zeros(X.shape[1])

    for train, test in cv.split(X, y):
        model = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        model.fit(X[train], y[train])
        prob = model.predict_proba(X[test])[:, 1]
        aucs.append(roc_auc_score(y[test], prob))
        importances += model.feature_importances_ / folds

    imp = pd.Series(importances, index=log_data.columns).sort_values(ascending=False)
    return ForestResult(aucs, imp)
