#!/bin/bash

RUN_DIR="data/runs/final_comparison_03/"
COMMON_ARGS="-or 0.9 -runs_dir $RUN_DIR --aumento_datos -b 16000 --lr 0.0005 -a CRATE_verysmall -e 1000"

# ==============================================================================
# para mirar cuantos tokens
# ==============================================================================


#python -m src.main -tp 50 -tt 10  $COMMON_ARGS
#python -m src.main -tp 60 -tt 10  $COMMON_ARGS
#python -m src.main -tp 70 -tt 10  $COMMON_ARGS
python -m src.main -tp 75 -tt 15  $COMMON_ARGS
python -m src.main -tp 80 -tt 15  $COMMON_ARGS
python -m src.main -tp 95 -tt 15  $COMMON_ARGS
python -m src.main -tp 100 -tt 20  $COMMON_ARGS
python -m src.main -tp 120 -tt 20  $COMMON_ARGS
python -m src.main -tp 140 -tt 20  $COMMON_ARGS


echo "--------REMATOUSE------"
