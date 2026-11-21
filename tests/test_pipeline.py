import numpy as np
import pandas as pd
import pytest

from biomarker import ml, stats
from biomarker.data import INTERNAL_STANDARD, make_lcms_dataset
from biomarker.preprocess import (
    impute_half_minimum,
    normalize_internal_standard,
    normalize_median,
    preprocess,
    zero_threshold,
)


# ---------- Korrektur für multiples Testen ----------

def test_benjamini_hochberg_bekanntes_beispiel():
    # Werte selbst nachgerechnet (gleiches Ergebnis wie R: p.adjust(p, "BH"))
    p = np.array([0.01, 0.04, 0.03, 0.005])
    q = stats.benjamini_hochberg(p)
    assert q == pytest.approx([0.02, 0.04, 0.04, 0.02])


def test_bh_ist_monoton_und_hoechstens_eins():
    rng = np.random.default_rng(1)
    p = rng.random(200)
    q = stats.benjamini_hochberg(p)
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)
    assert q.max() <= 1 and np.all(q >= p)


def test_bonferroni():
    assert stats.bonferroni([0.01, 0.5]) == pytest.approx([0.02, 1.0])


# ---------- Vorverarbeitung ----------

def test_zero_threshold_behaelt_feature_wenn_eine_gruppe_gut_ist():
    groups = pd.Series(["a", "a", "b", "b"])
    df = pd.DataFrame({
        "nur_in_b": [0, 0, 5, 6],   # in a fehlt es immer -> trotzdem behalten
        "ueberall_leer": [0, 1, 0, 0],
        "voll": [1, 2, 3, 4],
    })
    kept = zero_threshold(df, groups, max_zero_fraction=0.3)
    assert list(kept.columns) == ["nur_in_b", "voll"]


def test_halbes_minimum():
    df = pd.DataFrame({"x": [0.0, 4.0, 10.0]})
    assert impute_half_minimum(df)["x"].tolist() == [2.0, 4.0, 10.0]


def test_interner_standard_entfernt_verduennung():
    # Probe 2 ist doppelt so konzentriert -> nach der Normalisierung gleich
    df = pd.DataFrame({"f": [10.0, 20.0], "IS": [100.0, 200.0]})
    norm = normalize_internal_standard(df, "IS")
    assert norm["f"].iloc[0] == pytest.approx(norm["f"].iloc[1])
    assert "IS" not in norm.columns


def test_median_normalisierung():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 6.0], "c": [5.0, 10.0]})
    norm = normalize_median(df)
    assert norm.median(axis=1).iloc[0] == pytest.approx(norm.median(axis=1).iloc[1])


def test_preprocess_ergebnis_hat_keine_luecken():
    ds = make_lcms_dataset(n_per_group=10, n_features=50, seed=3)
    log_data = preprocess(ds.intensities, ds.groups, standard=INTERNAL_STANDARD)
    assert not log_data.isna().any().any()
    assert np.isfinite(log_data.to_numpy()).all()
    assert INTERNAL_STANDARD not in log_data.columns


# ---------- Ganze Pipeline auf synthetischen Daten ----------

@pytest.fixture(scope="module")
def processed():
    ds = make_lcms_dataset(n_per_group=30, n_features=200, n_biomarkers=10, effect=1.2, seed=7)
    log_data = preprocess(ds.intensities, ds.groups, standard=INTERNAL_STANDARD)
    return ds, log_data


def test_pipeline_findet_echte_biomarker(processed):
    ds, log_data = processed
    result = stats.differential_analysis(log_data, ds.groups, "patient", "control")
    hits = set(stats.significant(result).index)
    truth = set(ds.true_biomarkers) & set(log_data.columns)
    recall = len(hits & truth) / len(truth)
    precision = len(hits & truth) / max(len(hits), 1)
    assert recall >= 0.7, f"zu wenige gefunden: {recall:.2f}"
    assert precision >= 0.7, f"zu viele falsche Treffer: {precision:.2f}"


def test_random_forest_trennt_die_gruppen(processed):
    ds, log_data = processed
    forest = ml.random_forest_cv(log_data, ds.groups, positive="patient", folds=3)
    assert forest.mean_auc > 0.8
    # Die wichtigsten Features sollten zum Großteil echte Biomarker sein
    top = set(forest.importance.index[:10])
    assert len(top & set(ds.true_biomarkers)) >= 5


def test_ohne_unterschied_kaum_treffer():
    # Kein Effekt -> FDR-Kontrolle: fast keine signifikanten Features
    ds = make_lcms_dataset(n_per_group=20, n_features=200, effect=0.0, seed=11)
    log_data = preprocess(ds.intensities, ds.groups, standard=INTERNAL_STANDARD)
    result = stats.differential_analysis(log_data, ds.groups, "patient", "control")
    assert len(stats.significant(result)) <= 2


def test_pca_form(processed):
    ds, log_data = processed
    scores, var = ml.pca_scores(log_data, n_components=3)
    assert scores.shape == (len(log_data), 3)
    assert np.all(np.diff(var) <= 0)
