#!/bin/bash

# Test CRATE_enana with Linformer enabled
python -m src.main --aumento_datos -b 4096 -tp 75 -tt 15 -or 0.5 -runs_dir data/runs/test_linformer/ --arch CRATE_enana --paciencia 1000 --linformer

# Test CRATE_enana with Linformer disabled
python -m src.main --aumento_datos -b 4096 -tp 75 -tt 15 -or 0.5 -runs_dir data/runs/test_linformer/ --arch CRATE_enana --paciencia 1000

# Test CRATE_enana with shared_dict enabled
python -m src.main --aumento_datos -b 4096 -tp 75 -tt 15 -or 0.5 -runs_dir data/runs/test_linformer/ --arch CRATE_enana --paciencia 1000 --shared_dict

# Test CRATE_enana with no_pos enabled
python -m src.main --aumento_datos -b 4096 -tp 75 -tt 15 -or 0.5 -runs_dir data/runs/test_linformer/ --arch CRATE_enana --paciencia 1000 --no_pos

# Test CRATE_enana with second-order Neumann approximation
python -m src.main --aumento_datos -b 4096 -tp 75 -tt 15 -or 0.5 -runs_dir data/runs/test_linformer/ --arch CRATE_enana --paciencia 1000 --order second