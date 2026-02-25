#!/usr/bin/env bash

DIR="$1"

if [ -z "$DIR" ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

# Run plot_logs once on the directory
python -m src.plots.plot_logs "$DIR"

# Loop over files in DIR
for file in "$DIR"/*; do
  # Skip if not a regular file
  [ -f "$file" ] || continue

  # Extract filename without path and extension
  filename="$(basename -- "$file")"
  file_name="${filename%.*}"

  weight_path="data/weights/${file_name}.pth.tar"

  python -m src.evaluation.patch_inference "$weight_path" "$DIR"
  python -m src.evaluation.sparse_y_crate "$weight_path" "$DIR"
  python -m src.evaluation.mapas_atencion "$weight_path" "$DIR" -imaxes 12 -capas 4
done
