#!/bin/bash

# Test CRATE_enana with Linformer disabled
python -m src.main -t_dir data/RFMiD/Training_Set -v_dir data/RFMiD/Validation_Set --dataset rfmid --aumento_datos -b 40  -tp 1395 -tt 45 -or 0.5 -runs_dir data/runs/test_linformer/ --arch CRATE_enana --paciencia -1

# Test CRATE_enana with Linformer enabled
python -m src.main -t_dir data/RFMiD/Training_Set -v_dir data/RFMiD/Validation_Set --dataset rfmid --aumento_datos -b 40  -tp 1395 -tt 45 -or 0.5 -runs_dir data/runs/test_linformer/ --arch CRATE_enana --paciencia -1 --linformer

# Test CRATE_enana with shared_dict enabled
python -m src.main -t_dir data/RFMiD/Training_Set -v_dir data/RFMiD/Validation_Set --dataset rfmid --aumento_datos -b 40  -tp 1395 -tt 45 -or 0.5 -runs_dir data/runs/test_linformer/ --arch CRATE_enana --paciencia -1 --shared_dict --linformer


# Test CRATE_enana with second-order Neumann approximation
python -m src.main -t_dir data/RFMiD/Training_Set -v_dir data/RFMiD/Validation_Set --dataset rfmid --aumento_datos -b 40  -tp 1395 -tt 45 -or 0.5 -runs_dir data/runs/test_linformer/ --arch CRATE_enana --paciencia -1 --order second --linformer

# Test CRATE_enana with no_pos enabled
python -m src.main -t_dir data/RFMiD/Training_Set -v_dir data/RFMiD/Validation_Set --dataset rfmid --aumento_datos -b 40  -tp 1395 -tt 45 -or 0.5 -runs_dir data/runs/test_linformer/ --arch CRATE_enana --paciencia -1 --no_pos --linformer
