#!/usr/bin/env bash

DIR="$1"

if [ -z "$DIR" ]; then
  echo "Uso: $0 <directorio>"
  exit 1
fi

#python -m src.plots.plot_logs "$DIR"
#python -m src.evaluation.sparse_y_crate all "$DIR"

for file in "$DIR"/*; do
  echo "Estamos co archivo $file"
  [ -f "$file" ] || continue # pasamos si no es un archivo


  filename="$(basename -- "$file")"
  file_name="${filename%.*}"

  weight_path="data/weights/${file_name}.pth.tar"

  #python -m src.evaluation.patch_inference "$weight_path" "$DIR"
  python -m src.evaluation.patch_embeddin "$weight_path" "$DIR" -C -val
  #python -m src.evaluation.mapas_atencion "$weight_path" "$DIR"

done
