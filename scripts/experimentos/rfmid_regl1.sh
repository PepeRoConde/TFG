#!/bin/bash

penalties=(0.5 0.1 0.01 0.005 0.001 0.0005)

for p in "${penalties[@]}"; do
  python -m src.main \
    --aumento_datos \
    -b 40 \
    -tp 1395 \
    -tt 45 \
    -or 0.5 \
    -e 40000 \
    -lr 0.0001 \
    -runs_dir data/runs/penalization_embedding_05/ \
    --arch CRATE_enana \
    -t_dir data/RFMiD/Training_Set/ \
    -v_dir data/RFMiD/Validation_Set/ \
    --dataset rfmid \
    --embedding_l1_penalty "$p" \
    --order second \
    --shared_dict \
    --shared_u
done
