#!/bin/bash

# Test CRATE_enana with Linformer disabled
python -m src.main -t_dir data/RFMiD/Training_Set -v_dir data/RFMiD/Validation_Set --dataset rfmid --aumento_datos -b 16  -tp 1395 -tt 45 -or 0.5 -runs_dir data/runs/test_linformer/ --arch CRATE_tiny --paciencia 1000

# Test CRATE_enana with Linformer enabled
python -m src.main -t_dir data/RFMiD/Training_Set -v_dir data/RFMiD/Validation_Set --dataset rfmid --aumento_datos -b 40  -tp 1410 -tt 15 -or 0.5 -runs_dir data/runs/test_linformer/ --arch CRATE_tiny --paciencia 1000 --linformer 

# Test CRATE_enana with shared_dict enabled
python -m src.main -t_dir data/RFMiD/Training_Set -v_dir data/RFMiD/Validation_Set --dataset rfmid --aumento_datos -b 40  -tp 1410 -tt 15 -or 0.5 -runs_dir data/runs/test_linformer/ --arch CRATE_tiny --paciencia 1000 --shared_dict

# Test CRATE_enana with no_pos enabled
python -m src.main -t_dir data/RFMiD/Training_Set -v_dir data/RFMiD/Validation_Set --dataset rfmid --aumento_datos -b 40  -tp 1410 -tt 15 -or 0.5 -runs_dir data/runs/test_linformer/ --arch CRATE_tiny --paciencia 1000 --no_pos

# Test CRATE_enana with second-order Neumann approximation
python -m src.main -t_dir data/RFMiD/Training_Set -v_dir data/RFMiD/Validation_Set --dataset rfmid --aumento_datos -b 40  -tp 1410 -tt 15 -or 0.5 -runs_dir data/runs/test_linformer/ --arch CRATE_tiny --paciencia 1000 --order second
