#!/bin/bash

penalties=(0.5 0.1 0.01 0.005 0.001 0.0005)

common_args=(
  -tp 256
  -tt 16
  -or 0.0
  -e 25000
  -lr 0.00005
  -j 24
  --prefetch_factor 4
  -runs_dir data/runs/penalization_embedding_07/
  --arch CRATE_enana
  -t_dir data/RFMiD_256x256/Training_Set/
  -v_dir data/RFMiD_256x256/Validation_Set/
  --dataset rfmid
  --shared_dict
  --shared_u
  --print_freq 3
  --gain 2
  -b 2000
)

for p in "${penalties[@]}"; do
  python -m src.main \
    "${common_args[@]}" \
    --embedding_l1_penalty "$p" \
    --order second

  python -m src.main \
    "${common_args[@]}" \
    --embedding_l1_penalty "$p" \
    --order first

done
