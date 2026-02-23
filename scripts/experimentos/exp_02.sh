#!/bin/bash

RUN_DIR="data/runs/final_comparison_01/"
COMMON_ARGS="-or 0.6 -runs_dir $RUN_DIR --aumento_datos"

# ==============================================================================
# BEST KNOWN CONFIGS FROM ALL EXPERIMENTS:
#
# CRATE_tiny best: patch=14, token=7, label_smoothing=0.0, batch=8192
#   (ab2003 from tamano_04: val_acc ~95.4%, val_AUC ~0.967)
#
# CRATE_verysmall best: patch=80, token=16, label_smoothing=0.0, batch=512
#   (87f4f2 from stability_01: val_loss ~0.168, val_acc ~94.9%, val_AUC ~0.962)
#
# This script runs clean head-to-head comparisons of normal vs 2nd architectures
# ==============================================================================

echo "===== CRATE_tiny experiments ====="

echo "===== EXP 1: CRATE_tiny baseline (14/7, best known) ====="
python -m src.main -a CRATE_tiny -tp 14 -tt 7 -b 8192 --label_smoothing 0.0 $COMMON_ARGS

echo "===== EXP 2: CRATE_tiny variant 1 (16/8, competitive alternative) ====="
python -m src.main -a CRATE_tiny -tp 16 -tt 8 -b 8192 --label_smoothing 0.0 $COMMON_ARGS

echo "===== EXP 3: CRATE_tiny variant 2 (14/7 + class_weight=5.0) ====="
python -m src.main -a CRATE_tiny -tp 14 -tt 7 -b 8192 --label_smoothing 0.0 --class_weight 5.0 $COMMON_ARGS

echo "===== CRATE_tiny2nd experiments ====="

echo "===== EXP 4: CRATE_tiny2nd baseline (14/7, same as tiny EXP 1) ====="
python -m src.main -a CRATE_tiny2nd -tp 14 -tt 7 -b 8192 --label_smoothing 0.0 $COMMON_ARGS

echo "===== EXP 5: CRATE_tiny2nd variant 1 (16/8, same as tiny EXP 2) ====="
python -m src.main -a CRATE_tiny2nd -tp 16 -tt 8 -b 8192 --label_smoothing 0.0 $COMMON_ARGS

echo "===== EXP 6: CRATE_tiny2nd variant 2 (14/7 + class_weight=5.0) ====="
python -m src.main -a CRATE_tiny2nd -tp 14 -tt 7 -b 8192 --label_smoothing 0.0 --class_weight 5.0 $COMMON_ARGS

echo "===== ALL EXPERIMENTS FINISHED ====="
echo ""
echo "COMPARISON MATRIX:"
echo "  Config         | CRATE_tiny | CRATE_tiny2nd | Winner?"
echo "  14/7 baseline  | EXP 1      | EXP 4         | ?"
echo "  16/8 variant   | EXP 2      | EXP 5         | ?"
echo "  14/7 + cw=5.0  | EXP 3      | EXP 6         | ?"
echo ""
echo "Key questions:"
echo "1. Does tiny2nd close the gap with optimal hyperparams?"
echo "2. Does class_weight=5.0 help either architecture?"
echo "3. Is 14/7 or 16/8 the better patch/token configuration?"
