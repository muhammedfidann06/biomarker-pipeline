"""Die ganze Pipeline von den Rohdaten bis zu den Ergebnissen.

Beispiele:
    python run_pipeline.py                      # mit synthetischen Daten
    python run_pipeline.py --csv meine_daten.csv --group-col group --standard IS_d5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from biomarker import ml, plots, stats
from biomarker.data import INTERNAL_STANDARD, make_lcms_dataset
from biomarker.preprocess import preprocess


def load_csv(path: Path, group_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """CSV: eine Zeile pro Probe, eine Spalte mit der Gruppe, der Rest sind Features."""
    df = pd.read_csv(path, index_col=0)
    groups = df.pop(group_col)
    return df, groups


def main() -> None:
    ap = argparse.ArgumentParser(description="LC-MS Biomarker-Pipeline")
    ap.add_argument("--csv", type=Path, help="eigene Daten (sonst synthetisch)")
    ap.add_argument("--group-col", default="group")
    ap.add_argument("--standard", default=None, help="Spalte mit dem internen Standard")
    ap.add_argument("--case", default="patient")
    ap.add_argument("--control", default="control")
    ap.add_argument("--imputation", choices=["half_min", "knn"], default="half_min")
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    truth: list[str] = []
    if args.csv:
        raw, groups = load_csv(args.csv, args.group_col)
        standard = args.standard
    else:
        ds = make_lcms_dataset(seed=42)
        raw, groups, truth = ds.intensities, ds.groups, ds.true_biomarkers
        standard = INTERNAL_STANDARD
        print("Keine CSV angegeben -> synthetische Daten mit bekannten Biomarkern.")

    args.out.mkdir(parents=True, exist_ok=True)

    # s01-s03: Filtern, Imputation, Normalisierung, log2
    log_data = preprocess(raw, groups, standard=standard, imputation=args.imputation)
    print(f"Features: {raw.shape[1]} roh -> {log_data.shape[1]} nach dem Filtern, Proben: {len(log_data)}")

    # s04-s06: Test pro Feature + Korrektur
    result = stats.differential_analysis(log_data, groups, args.case, args.control)
    hits = stats.significant(result, q=args.q)
    result.to_csv(args.out / "differential_analysis.csv")
    print(f"Test: {result.attrs['test']}, signifikant (q < {args.q}): {len(hits)}")

    # s07: Abbildungen
    plots.volcano(result, args.out / "volcano.png", q=args.q)
    scores, variance = ml.pca_scores(log_data)
    plots.pca_plot(scores, groups, variance, args.out / "pca.png")
    top = list(hits.index[:6]) or list(result.index[:6])
    plots.top_boxplots(log_data, groups, top, args.out / "top_features.png")

    # s08: Random Forest – wie gut trennen die Daten die Gruppen?
    forest = ml.random_forest_cv(log_data, groups, positive=args.case)
    forest.importance.head(30).to_csv(args.out / "rf_importance.csv", header=["importance"])
    print(f"Random Forest AUC (5-fach CV): {forest.mean_auc:.3f}")
    plots.roc_plot(forest.y_true, forest.oof_probability, args.out / "roc.png")

    summary = {
        "features_raw": int(raw.shape[1]),
        "features_after_filter": int(log_data.shape[1]),
        "test": result.attrs["test"],
        "significant": len(hits),
        "rf_mean_auc": round(forest.mean_auc, 4),
    }
    if truth:
        # Nur bei synthetischen Daten: Wie viele echte Biomarker haben wir gefunden?
        found = set(hits.index) & set(truth)
        summary["true_biomarkers"] = len(truth)
        summary["recall"] = round(len(found) / len(truth), 3)
        summary["precision"] = round(len(found) / max(len(hits), 1), 3)
        print(f"Echte Biomarker gefunden: {len(found)}/{len(truth)}, Precision {summary['precision']:.2f}")

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Ergebnisse in {args.out}/")


if __name__ == "__main__":
    main()
