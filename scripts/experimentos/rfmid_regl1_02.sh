#!/bin/bash

penalties=(0.5 0.1 0.01 0.005 0.001 0.0005)

common_args=(
  -tp 512
  -tt 16
  -or 0.0
  -e 40000
  -lr 0.00001
  -j 24
  --prefetch_factor 4
  -runs_dir data/runs/penalization_embedding_06/
  --arch CRATE_enana
  -t_dir data/RFMiD_512x512/Training_Set/
  -v_dir data/RFMiD_512x512/Validation_Set/
  --dataset rfmid
  --shared_dict
  --shared_u
  --print_freq 3
  --gain 2
)

for p in "${penalties[@]}"; do
  python -m src.main \
    -b 209 \
    "${common_args[@]}" \
    --embedding_l1_penalty "$p" \
    --order second

  python -m src.main \
    -b 375 \
    "${common_args[@]}" \
    --embedding_l1_penalty "$p" \
    --order first

done
