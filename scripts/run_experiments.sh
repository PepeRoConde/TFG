python -m src.main --aumento_datos -j 16 --wd 0.100 --lr 0.00500 -b 8192 -tp 25 -tt 5 -e 500 --label_mode vainilla --dataset online --label_smoothing 0.0 -or 0.8
python -m src.main --aumento_datos -j 16 --wd 0.010 --lr 0.00500 -b 8192 -tp 25 -tt 5 -e 500 --label_mode vainilla --dataset online --label_smoothing 0.0 -or 0.8
python -m src.main --aumento_datos -j 16 --wd 0.001 --lr 0.00500 -b 8192 -tp 25 -tt 5 -e 500 --label_mode vainilla --dataset online --label_smoothing 0.0 -or 0.8
python -m src.main --aumento_datos -j 16 --wd 0.100 --lr 0.00050 -b 8192 -tp 25 -tt 5 -e 500 --label_mode vainilla --dataset online --label_smoothing 0.0 -or 0.8 
python -m src.main --aumento_datos -j 16 --wd 0.010 --lr 0.00050 -b 8192 -tp 25 -tt 5 -e 500 --label_mode vainilla --dataset online --label_smoothing 0.0 -or 0.8 
python -m src.main --aumento_datos -j 16 --wd 0.001 --lr 0.00050 -b 8192 -tp 25 -tt 5 -e 500 --label_mode vainilla --dataset online --label_smoothing 0.0 -or 0.8 
python -m src.main --aumento_datos -j 16 --wd 0.100 --lr 0.00005 -b 8192 -tp 25 -tt 5 -e 500 --label_mode vainilla --dataset online --label_smoothing 0.0 -or 0.8 
python -m src.main --aumento_datos -j 16 --wd 0.010 --lr 0.00005 -b 8192 -tp 25 -tt 5 -e 500 --label_mode vainilla --dataset online --label_smoothing 0.0 -or 0.8 
python -m src.main --aumento_datos -j 16 --wd 0.001 --lr 0.00005 -b 8192 -tp 25 -tt 5 -e 500 --label_mode vainilla --dataset online --label_smoothing 0.0 -or 0.8 
#python -m src.main --aumento_datos -j 16 -b 2048 -tp 256 -tt 4 -or 0.1 -e 500 -s 1
#python -m src.main --aumento_datos -j 16 -b 2048 -tp 256 -tt 32 -or 0.1 -e 500 -s 2
#python -m src.main --aumento_datos -j 16 -b 2048 -tp 128 -tt 4 -or 0.1 -e 500 -s 1
#python -m src.main --aumento_datos -j 16 -b 2048 -tp 128 -tt 32 -or 0.1 -e 500 -s 2
#python -m src.main --aumento_datos -j 16 -b 2048 -tp 80 -tt 16 -or 0.1 -e 500 --label_mode gaussian -s 1 
#python -m src.main --aumento_datos -j 16 -b 2048 -tp 80 -tt 16 -or 0.1 -e 500 --label_mode gaussian -s 1.41
#python -m src.main --aumento_datos -j 16 -b 2048 -tp 80 -tt 16 -or 0.1 -e 500 --label_mode gaussian -s 2 
#python -m src.main --aumento_datos -j 16 -b 2048 -tp 80 -tt 16 -or 0.1 -e 500 --label_mode gaussian -s 4 
#python -m src.main --aumento_datos -j 16 -b 2048 -tp 15 -tt 5  -or 0.1 -e 500 --label_mode vainilla
#python -m src.main --aumento_datos -j 16 -b 2048 -tp 25 -tt 5  -or 0.1 -e 500 --label_mode vainilla 
#python -m src.main --aumento_datos -j 16 -b 2048 -tp 25 -tt 5  -or 0.1 -e 500 --label_mode vainilla 
#python -m src.main --aumento_datos -j 16 -b 2048 -tp 30 -tt 10 -or 0.1 -e 500 --label_mode vainilla 
#python -m src.main --aumento_datos -j 16 -b 2048 -tp 30 -tt 10 -or 0.1 -e 500 --label_mode vainilla 
#python -m src.main --aumento_datos -j 16 -b 2048 -tp 50 -tt 10 -or 0.1 -e 500 --label_mode vainilla 
#python -m src.main --aumento_datos -j 16 -b 2048 -tp 50 -tt 10 -or 0.1 -e 500 --label_mode vainilla 
#python -m src.main --aumento_datos -j 16 -b 2048 -tp 50 -tt 10 -or 0.1 -e 500 --label_mode vainilla 
#python -m src.main --aumento_datos -j 16 -b 2048 -tp 50 -tt 10 -or 0.1 -e 500 --label_mode vainilla 
#python -m src.main --aumento_datos -j 16 -b 2048 -tp 80 -tt 16 -or 0.1 -e 500 --label_mode vainilla 
#python -m src.main --aumento_datos -j 16 -b 2048 -tp 80 -tt 16 -or 0.1 -e 500 --label_mode vainilla 
#python -m src.main --data_augmentation -j 7 -b 2048 -tp 48 -tt 16 -e 300 -t_dir data/DRIVE/train_patches_48 -v_dir data/DRIVE/val_patches_48 -s 1
#python -m src.main --data_augmentation -j 7 -b 2048 -tp 48 -tt 16 -e 300 -t_dir data/DRIVE/train_patches_48 -v_dir data/DRIVE/val_patches_48 -s 2
#python -m src.main --data_augmentation -j 7 -b 2048 -tp 48 -tt 16 -e 300 -t_dir data/DRIVE/train_patches_48 -v_dir data/DRIVE/val_patches_48 -s 4
#python -m src.main --data_augmentation -j 7 -b 2048 -tp 48 -tt 16 -e 300 -t_dir data/DRIVE/train_patches_48 -v_dir data/DRIVE/val_patches_48 -s 1.41
#python main.py --epochs 100 --lr 0.0001 --tamano_patch 48 --tamano_token 16 --label_mode gaussian --sigma 0.7	
#python main.py --epochs 100 --lr 0.0001 --tamano_patch 36 --tamano_token 12 --label_mode gaussian --sigma 0.7
#python main.py --epochs 100 --lr 0.0001 --tamano_patch 64 --tamano_token 16 --label_mode gaussian --sigma 0.7
#python main.py --epochs 100 --lr 0.0001 --tamano_patch 30 --tamano_token 10 --label_mode gaussian --sigma 0.7
#python main.py --epochs 100 --lr 0.0001 --tamano_patch 60 --tamano_token 20 --label_mode gaussian --sigma 0.7
#python main.py --epochs 30 --tamano_patch 48 --tamano_token 16 --label_mode gaussian --sigma 0.1
#python main.py --epochs 30 --tamano_patch 48 --tamano_token 16 --label_mode gaussian --sigma 0.2
#python main.py --epochs 30 --tamano_patch 48 --tamano_token 16 --label_mode gaussian --sigma 0.3
#python main.py --epochs 30 --tamano_patch 48 --tamano_token 16 --label_mode gaussian --sigma 0.5
#python main.py --epochs 30 --tamano_patch 48 --tamano_token 16 --label_mode gaussian --sigma 0.7
#python main.py --epochs 30 --tamano_patch 48 --tamano_token 16 --label_mode gaussian --sigma 1
#python main.py --epochs 30 --tamano_patch 48 --tamano_token 16 --label_mode gaussian --sigma 2
#python main.py --epochs 30 --tamano_patch 24 --tamano_token 8 --label_mode gaussian --sigma 5
