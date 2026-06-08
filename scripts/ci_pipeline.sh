#!/usr/bin/env bash
# scripts/ci_pipeline.sh
# Reduced pipeline for CI (and local dry-runs). Skips eda + experiment grid
# (not required by any guard test) to keep runtime under 5 minutes.
# H2O, CodeCarbon, and Evidently Cloud are already try/except-wrapped and
# skip cleanly when unavailable.
set -e

echo "[1/8] Data Lineage (Raw→Mart) .........."; uv run python -m src.lineage.run_lineage
echo "[2/8] Preprocess + split ................"; uv run python -m src.preprocessing
echo "[3/8] Feature Store (v1 + v2) ..........."; uv run python -m src.feature_store
echo "[4/8] AutoML Benchmark (FLAML only) ....."; uv run python -m src.automl_benchmark --flaml-budget 15
echo "[5/8] Evaluate (original test) .........."; uv run python -m src.evaluate
echo "[6/8] Build modified test set ..........."; uv run python -m src.make_modified_test
echo "[7/8] PSI drift monitoring .............."; uv run python -m src.monitoring
echo "[8/8] Evidently drift report ............"; uv run python -m src.monitoring_evidently

echo ""
echo "ci_pipeline.sh done — all artifacts generated."
