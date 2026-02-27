#!/usr/bin/env bash
# Scale-up pipeline for top-300 S&P tickers (100 trials per model)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Windows cp949 肄섏넄?먯꽌???대え吏/?좊땲肄붾뱶 異쒕젰?쇰줈 以묐떒?섏? ?딅룄濡?UTF-8 媛뺤젣
export PYTHONUTF8=1
export PYTHONIOENCODING=UTF-8

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "Python executable not found (python3/python)."
    exit 1
fi

echo "=========================================================="
echo "STARTING 300-TICKER ML & MAPPING PIPELINE"
echo "Project Root: ${PROJECT_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Start Time: $(date)"
echo "=========================================================="

echo ""
echo "[1/9] Downloading Raw Stock Data from Yahoo Finance..."
"${PYTHON_BIN}" -u -m DB.update_stock_data --mode auto

echo ""
echo "[2/9] Calculating Log Returns for All Tickers..."
"${PYTHON_BIN}" -u -m DB.calculate_log_returns

echo ""
echo "[3/9] Updating Market Features (VIX, DXY)..."
"${PYTHON_BIN}" -u -m DB.update_market_data

echo ""
echo "[4/9] Updating S&P 500 Benchmark Data..."
"${PYTHON_BIN}" -u -m DB.update_sp500_data

echo ""
echo "[5/9] Calculating EWMA Covariance Matrix..."
"${PYTHON_BIN}" -u -m DB.calculate_ewma

echo ""
echo "[6/9] Running Unified MASTER_FEATURES Generation (Panel Data)..."
"${PYTHON_BIN}" -u -m DB.build_master_dataset

echo ""
echo "[7/9] Running ML Pipeline with 100 Trials per Model (Phase 3)..."
echo "      (This step will take a long time!)"
"${PYTHON_BIN}" -u -m DB.run_all_tickers --force-retune-all --xgb-trials 100 --svm-trials 100 --rf-trials 100 --logreg-trials 100

echo ""
echo "[8/9] Running Mapping & Final Aggregation (Phase 4 & 5)..."
"${PYTHON_BIN}" -u -m DB.run_all_mapping

echo ""
echo "      -> Applying Regime Shift Overlay (Graduated Momentum Tracking)..."
"${PYTHON_BIN}" -u -m DB.adjust_regime
if [ $? -ne 0 ]; then
    echo "??Error in Regime Shift Overlay. Exiting."
    exit 1
fi

echo ""
echo "[9/9] Syncing Risk-Level Portfolio Snapshot..."
"${PYTHON_BIN}" -u -m DB.update_risk_level_portfolio_snapshot

echo ""
echo "=========================================================="
echo "PIPELINE COMPLETE!"
echo "End Time: $(date)"
echo "=========================================================="

