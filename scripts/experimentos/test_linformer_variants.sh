#!/bin/bash

# Test CRATE_enana with Linformer enabled and different share_proj values
python -m src.main --aumento_datos -b 4096 -tp 75 -tt 15 -or 0.5 -runs_dir data/runs/test_linformer_variants/ --arch CRATE_enana --paciencia 1000 --linformer --share_proj none
python -m src.main --aumento_datos -b 4096 -tp 75 -tt 15 -or 0.5 -runs_dir data/runs/test_linformer_variants/ --arch CRATE_enana --paciencia 1000 --linformer --share_proj headwise
python -m src.main --aumento_datos -b 4096 -tp 75 -tt 15 -or 0.5 -runs_dir data/runs/test_linformer_variants/ --arch CRATE_enana --paciencia 1000 --linformer --share_proj key-value
python -m src.main --aumento_datos -b 4096 -tp 75 -tt 15 -or 0.5 -runs_dir data/runs/test_linformer_variants/ --arch CRATE_enana --paciencia 1000 --linformer --share_proj layerwise

# Test CRATE_enana with Linformer enabled and different project_dim values
python -m src.main --aumento_datos -b 4096 -tp 75 -tt 15 -or 0.5 -runs_dir data/runs/test_linformer_variants/ --arch CRATE_enana --paciencia 1000 --linformer --project_dim 64
python -m src.main --aumento_datos -b 4096 -tp 75 -tt 15 -or 0.5 -runs_dir data/runs/test_linformer_variants/ --arch CRATE_enana --paciencia 1000 --linformer --project_dim 128
python -m src.main --aumento_datos -b 4096 -tp 75 -tt 15 -or 0.5 -runs_dir data/runs/test_linformer_variants/ --arch CRATE_enana --paciencia 1000 --linformer --project_dim 256
python -m src.main --aumento_datos -b 4096 -tp 75 -tt 15 -or 0.5 -runs_dir data/runs/test_linformer_variants/ --arch CRATE_enana --paciencia 1000 --linformer --project_dim 512