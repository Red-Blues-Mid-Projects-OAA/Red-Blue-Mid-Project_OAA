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
echo "[1/3] Running Master Dataset Generation (Phase 2)..."
python3 -u scripts/build_all_master_datasets.py
if [ $? -ne 0 ]; then
    echo "❌ Error in Master Dataset Generation. Exiting."
    exit 1
fi

echo ""
echo "[2/3] Running ML Pipeline with 100 Trials per Model (Phase 3)..."
echo "      (This step will take a long time!)"
python3 -u -m Classification.multi_ticker.run_all_tickers --force-retune-all --xgb-trials 100 --svm-trials 100 --rf-trials 100 --logreg-trials 100
if [ $? -ne 0 ]; then
    echo "❌ Error in ML Pipeline. Exiting."
    exit 1
fi

echo ""
echo "[3/3] Running Mapping & Final Aggregation (Phase 4 & 5)..."
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
