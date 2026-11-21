# 🧪 Biomarker Discovery Pipeline (LC-MS)

[![ci](https://github.com/muhammedfidann06/biomarker-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/muhammedfidann06/biomarker-pipeline/actions/workflows/ci.yml)

A reproducible Python pipeline for finding biomarkers in LC-MS data (metabolomics / peptides): from raw intensities to statistically significant features and a machine-learning model.

## Pipeline
| Step | What happens | Module |
|---|---|---|
| s01 Zero thresholding | drop features with too many zeros – **per group**, so group-specific markers survive | `preprocess.zero_threshold` |
| s02 Imputation | half-minimum (default) or k-nearest neighbours | `preprocess.impute_*` |
| s03 Normalization | internal standard (IS) or median normalization, then log2 | `preprocess.normalize_*` |
| s04 Normality | Shapiro–Wilk per feature and group → chooses Welch t-test or Mann–Whitney U | `stats.normality` |
| s05 Testing | per-feature test, log2 fold change; Kruskal–Wallis for ≥ 3 groups | `stats.differential_analysis` |
| s06 Correction | Benjamini–Hochberg (FDR) and Bonferroni – implemented from scratch | `stats.benjamini_hochberg` |
| s07 Visualization | volcano plot, PCA, boxplots of the top features | `plots` |
| s08 Machine learning | Random Forest with stratified 5-fold CV, ROC AUC, feature importance; ROC curve from out-of-fold predictions | `ml.random_forest_cv`, `plots.roc_plot` |

## Validation with synthetic data
Real clinical data is usually not public, so `biomarker/data.py` generates realistic LC-MS data: log-normal intensities, a per-sample dilution effect, values below the detection limit and a set of **known** biomarkers. The tests check that the pipeline
- finds at least 70 % of the true biomarkers with at least 70 % precision,
- finds (almost) nothing when there is no real effect (FDR control),
- removes the dilution effect with the internal standard,
- and that the Random Forest separates the groups (AUC > 0.8) and ranks true markers highest.

## Run
```bash
pip install -r requirements.txt
pytest                      # tests
python run_pipeline.py      # synthetic demo -> results/
python run_pipeline.py --csv my_data.csv --group-col group --standard IS_d5 --case patient --control control
```
Output in `results/`: `differential_analysis.csv`, `rf_importance.csv`, `summary.json`, `volcano.png`, `pca.png`, `top_features.png`, `roc.png`.
The CI runs the tests on Python 3.10 and 3.12 and uploads the results as an artifact.

## Kurzbeschreibung (Deutsch)
Eine Pipeline für LC-MS-Daten: Filtern, fehlende Werte auffüllen, normalisieren, statistisch testen
(mit FDR-Korrektur) und ein Random-Forest-Modell. Mit synthetischen Daten prüfen die Tests,
ob die Pipeline die echten Biomarker wirklich findet.
