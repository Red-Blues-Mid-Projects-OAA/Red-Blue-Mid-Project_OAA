#!/bin/bash
# Scale-Up Pipeline Run for 300 S&P 500 Tickers
# 100 Trials per Model for Hyperparameter Tuning

PROJECT_ROOT="/Users/wowjd/Desktop/Private/授業/Project-2_OAA"
cd $PROJECT_ROOT

echo "=========================================================="
echo "🚀 STARTING 300-TICKER ML & MAPPING PIPELINE 🚀"
echo "Start Time: $(date)"
echo "=========================================================="

echo ""
echo "[0/5] Downloading Raw Stock Data from Yahoo Finance..."
python3 -u -m DB.update_stock_data --mode auto
if [ $? -ne 0 ]; then
    echo "❌ Error in Downloading Raw Stock Data. Exiting."
    exit 1
fi

echo ""
echo "[1/5] Calculating Log Returns for All Tickers..."
python3 -u -m DB.calculate_log_returns
if [ $? -ne 0 ]; then
    echo "❌ Error in Calculating Log Returns. Exiting."
    exit 1
fi

echo ""
echo "[2/5] Updating Market Features (VIX, DXY)..."
python3 -u -m DB.update_market_data
if [ $? -ne 0 ]; then
    echo "❌ Error in Updating Market Features. Exiting."
    exit 1
fi

echo ""
echo "[3/5] Updating S&P 500 Benchmark Data..."
python3 -u -m DB.update_sp500_data
if [ $? -ne 0 ]; then
    echo "❌ Error in Updating S&P 500 Data. Exiting."
    exit 1
fi

echo ""
echo "[4/5] Calculating EWMA Covariance Matrix..."
python3 -u -m DB.calculate_ewma
if [ $? -ne 0 ]; then
    echo "❌ Error in Calculating EWMA. Exiting."
    exit 1
fi

echo ""
echo "[5/5] Running Master Dataset Generation (Feature Engineering)..."
python3 -u scripts/build_all_master_datasets.py
if [ $? -ne 0 ]; then
    echo "❌ Error in Master Dataset Generation. Exiting."
    exit 1
fi

echo ""
echo "[6/5] Running ML Pipeline with 100 Trials per Model (Phase 3)..."
echo "      (This step will take a long time!)"
python3 -u -m Classification.multi_ticker.run_all_tickers --force-retune-all --xgb-trials 100 --svm-trials 100 --rf-trials 100 --logreg-trials 100
if [ $? -ne 0 ]; then
    echo "❌ Error in ML Pipeline. Exiting."
    exit 1
fi

echo ""
echo "[7/5] Running Mapping & Final Aggregation (Phase 4 & 5)..."
python3 -u scripts/run_all_mapping.py
if [ $? -ne 0 ]; then
    echo "❌ Error in Final Mapping Aggregation. Exiting."
    exit 1
fi

echo ""
echo "=========================================================="
echo "🎉 PIPELINE COMPLETE! 🎉"
echo "End Time: $(date)"
echo "=========================================================="
