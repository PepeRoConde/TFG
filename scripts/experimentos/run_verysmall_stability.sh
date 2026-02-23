#!/bin/bash

RUN_DIR="data/runs/verysmall_stability_02/"
COMMON_ARGS="-tp 80 -tt 16 -or 0.9 -b 8192 -runs_dir $RUN_DIR"

echo "===== EXP 1: VerySmall baseline ====="
python -m src.main -a CRATE_verysmall --aumento_datos $COMMON_ARGS

echo "===== EXP 2: VerySmall 2nd order ====="
python -m src.main -a CRATE_verysmall2nd --aumento_datos $COMMON_ARGS

echo "===== EXP 5: No augmentation ====="
python -m src.main -a CRATE_verysmall $COMMON_ARGS

echo "===== EXP 6: Lower LR (2e-5) ====="
python -m src.main -a CRATE_verysmall --aumento_datos --lr 2e-5 $COMMON_ARGS

echo "===== EXP 7: Higher LR (1e-4) ====="
python -m src.main -a CRATE_verysmall --aumento_datos --lr 1e-4 $COMMON_ARGS

echo "===== EXP 8: No label smoothing ====="
python -m src.main -a CRATE_verysmall --aumento_datos --label_smoothing 0.0 $COMMON_ARGS

echo "===== EXP 9: Lion optimizer ====="
python -m src.main -a CRATE_verysmall --aumento_datos --optimizer Lion $COMMON_ARGS


echo "===== ALL EXPERIMENTS FINISHED ====="
