#!/bin/bash

RUN_DIR="data/runs/tamano_search_04/"
COMMON_ARGS="-tt 7 -or 0.6 -runs_dir $RUN_DIR --aumento_datos -b 8192 --label_smoothing 0.0"

# --- BASELINE (replicate c369c1 with label_smoothing=0.0) ---
echo "===== EXP 1: Baseline 21/7, no label smoothing ====="
python -m src.main -a CRATE_tiny -tp 21 $COMMON_ARGS

# --- PUSH SMALLER: does the trend continue below patch=21? ---
echo "===== EXP 2: patch=14, token=7 ====="
python -m src.main -a CRATE_tiny -tp 14 $COMMON_ARGS

echo "===== EXP 3: patch=16, token=8 ====="
python -m src.main -a CRATE_tiny -tp 16 -tt 8 -or 0.6 -runs_dir $RUN_DIR --aumento_datos -b 8192 --label_smoothing 0.0

# --- CONFIRM ratio=3 is the right ratio at small scale ---
echo "===== EXP 4: patch=18, token=6 ====="
python -m src.main -a CRATE_tiny -tp 18 -tt 6 -or 0.6 -runs_dir $RUN_DIR --aumento_datos -b 8192 --label_smoothing 0.0

echo "===== EXP 5: patch=21, token=3 (ratio=7) ====="
python -m src.main -a CRATE_tiny -tp 21 -tt 3 -or 0.6 -runs_dir $RUN_DIR --aumento_datos -b 8192 --label_smoothing 0.0

# --- BEST CONFIG + lessons from stability_01 ---
echo "===== EXP 6: patch=21, token=7, label_smoothing=0.1 (confirm 0.0 wins) ====="
python -m src.main -a CRATE_tiny -tp 21 -tt 7 -or 0.6 -runs_dir $RUN_DIR --aumento_datos -b 8192 --label_smoothing 0.1

echo "===== ALL EXPERIMENTS FINISHED ====="
