#!/usr/bin/env bash

DIR="$1"

if [ -z "$DIR" ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

python -m src.plots.plot_logs "$DIR"
python -m src.evaluation.sparse_y_crate all "$DIR"

for file in "$DIR"/*; do
  [ -f "$file" ] || continue # pasamos si no es un archivo

  filename="$(basename -- "$file")"
  file_name="${filename%.*}"

  weight_path="data/weights/${file_name}.pth.tar"

  #python -m src.evaluation.patch_inference "$weight_path" "$DIR"
  python -m src.evaluation.patch_embeddin "$weight_path" "$DIR" -imaxes 12 -k 12 -C
  #python -m src.evaluation.mapas_atencion "$weight_path" "$DIR" -imaxes 12 -capas 4
done
