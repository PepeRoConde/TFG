python -m src.main --aumento_datos -b 8192 -tp 80 -tt 16 -or 0.6 -runs_dir data/runs/1vs2_02/ --arch CRATE_tiny
python -m src.main --aumento_datos -b 8192 -tp 80 -tt 16 -or 0.6 -runs_dir data/runs/1vs2_02/ --arch CRATE_tiny2nd
python -m src.main --aumento_datos -b 8192 -tp 96 -tt 16 -or 0.6 -runs_dir data/runs/1vs2_02/ --arch CRATE_tiny
python -m src.main --aumento_datos -b 8192 -tp 96 -tt 16 -or 0.6 -runs_dir data/runs/1vs2_02/ --arch CRATE_tiny2nd

python -m src.main --aumento_datos -b 8192 -tp 48 -tt 16 -or 0.6 -runs_dir data/runs/tamano_search_02/
python -m src.main --aumento_datos -b 8192 -tp 64 -tt 16 -or 0.6 -runs_dir data/runs/tamano_search_02/
python -m src.main --aumento_datos -b 8192 -tp 80 -tt 16 -or 0.6 -runs_dir data/runs/tamano_search_02/
python -m src.main --aumento_datos -b 8192 -tp 96 -tt 16 -or 0.6 -runs_dir data/runs/tamano_search_02/
python -m src.main --aumento_datos -b 8192 -tp 30 -tt 10 -or 0.6 -runs_dir data/runs/tamano_search_02/
python -m src.main --aumento_datos -b 8192 -tp 40 -tt 10 -or 0.6 -runs_dir data/runs/tamano_search_02/
python -m src.main --aumento_datos -b 8192 -tp 50 -tt 10 -or 0.6 -runs_dir data/runs/tamano_search_02/
python -m src.main --aumento_datos -b 8192 -tp 60 -tt 16 -or 0.6 -runs_dir data/runs/tamano_search_02/
python -m src.main --aumento_datos -b 8192 -tp 21 -tt 7  -or 0.6 -runs_dir data/runs/tamano_search_02/
python -m src.main --aumento_datos -b 8192 -tp 28 -tt 7  -or 0.6 -runs_dir data/runs/tamano_search_02/
python -m src.main --aumento_datos -b 8192 -tp 35 -tt 7  -or 0.6 -runs_dir data/runs/tamano_search_02/
python -m src.main --aumento_datos -b 8192 -tp 42 -tt 7  -or 0.6 -runs_dir data/runs/tamano_search_02/
python -m src.main --aumento_datos -b 8192 -tp 49 -tt 7  -or 0.6 -runs_dir data/runs/tamano_search_02/



#python -m src.main --aumento_datos --wd 0.100 --lr 0.00050 -b 8192 -tp 80 -tt 16  -e 500 --label_mode vainilla --dataset online --label_smoothing 0.05 -or 0.9  --arch CRATE_tiny2nd -runs_dir data/runs/1vs2/
#python -m src.main --aumento_datos --wd 0.100 --lr 0.00050 -b 8192 -tp 80 -tt 16  -e 500 --label_mode vainilla --dataset online --label_smoothing 0.05 -or 0.9  --arch CRATE_tiny -runs_dir data/runs/1vs2/
#
#python -m src.main --aumento_datos --wd 0.100 --lr 0.00500 -b 8192 -tp 80 -tt 16  -e 500 --label_mode vainilla --dataset online --label_smoothing 0.0 -or 0.8 -runs_dir data/runs/wd-vs-lr/
#python -m src.main --aumento_datos --wd 0.010 --lr 0.00500 -b 8192 -tp 80 -tt 16  -e 500 --label_mode vainilla --dataset online --label_smoothing 0.0 -or 0.8 -runs_dir data/runs/wd-vs-lr/
#python -m src.main --aumento_datos --wd 0.001 --lr 0.00500 -b 8192 -tp 80 -tt 16  -e 500 --label_mode vainilla --dataset online --label_smoothing 0.0 -or 0.8 -runs_dir data/runs/wd-vs-lr/
#python -m src.main --aumento_datos --wd 0.100 --lr 0.00050 -b 8192 -tp 80 -tt 16  -e 500 --label_mode vainilla --dataset online --label_smoothing 0.0 -or 0.8 -runs_dir data/runs/wd-vs-lr/ 
#python -m src.main --aumento_datos --wd 0.010 --lr 0.00050 -b 8192 -tp 80 -tt 16  -e 500 --label_mode vainilla --dataset online --label_smoothing 0.0 -or 0.8 -runs_dir data/runs/wd-vs-lr/ 
#python -m src.main --aumento_datos --wd 0.001 --lr 0.00050 -b 8192 -tp 80 -tt 16  -e 500 --label_mode vainilla --dataset online --label_smoothing 0.0 -or 0.8 -runs_dir data/runs/wd-vs-lr/ 
#python -m src.main --aumento_datos --wd 0.100 --lr 0.00005 -b 8192 -tp 80 -tt 16  -e 500 --label_mode vainilla --dataset online --label_smoothing 0.0 -or 0.8 -runs_dir data/runs/wd-vs-lr/ 
#python -m src.main --aumento_datos --wd 0.010 --lr 0.00005 -b 8192 -tp 80 -tt 16  -e 500 --label_mode vainilla --dataset online --label_smoothing 0.0 -or 0.8 -runs_dir data/runs/wd-vs-lr/ 
#python -m src.main --aumento_datos --wd 0.001 --lr 0.00005 -b 8192 -tp 80 -tt 16  -e 500 --label_mode vainilla --dataset online --label_smoothing 0.0 -or 0.8 -runs_dir data/runs/wd-vs-lr/ 

#python -m src.main --aumento_datos -b 2048 -tp 256 -tt 4 -or 0.1 -e 500 -s 1
#python -m src.main --aumento_datos -b 2048 -tp 256 -tt 32 -or 0.1 -e 500 -s 2
#python -m src.main --aumento_datos -b 2048 -tp 128 -tt 4 -or 0.1 -e 500 -s 1
#python -m src.main --aumento_datos -b 2048 -tp 128 -tt 32 -or 0.1 -e 500 -s 2
#python -m src.main --aumento_datos -b 8192 -tp 80 -tt 16 -or 0.1 -e 500 -runs_dir data/runs/sigma --label_mode gaussian -s 1 
#python -m src.main --aumento_datos -b 8192 -tp 80 -tt 16 -or 0.1 -e 500 -runs_dir data/runs/sigma --label_mode gaussian -s 1.26 # 2^(1/3)  
#python -m src.main --aumento_datos -b 8192 -tp 80 -tt 16 -or 0.1 -e 500 -runs_dir data/runs/sigma --label_mode gaussian -s 1.41 # 2^(1/2)
#python -m src.main --aumento_datos -b 8192 -tp 80 -tt 16 -or 0.1 -e 500 -runs_dir data/runs/sigma --label_mode gaussian -s 2    # 2
#python -m src.main --aumento_datos -b 8192 -tp 80 -tt 16 -or 0.1 -e 500 -runs_dir data/runs/sigma --label_mode gaussian -s 4    # 2^2

#python -m src.main --aumento_datos -b 2048 -tp 15 -tt 5  -or 0.1 -e 500 --label_mode vainilla
#python -m src.main --aumento_datos -b 2048 -tp 25 -tt 5  -or 0.1 -e 500 --label_mode vainilla 
#python -m src.main --aumento_datos -b 2048 -tp 25 -tt 5  -or 0.1 -e 500 --label_mode vainilla 
#python -m src.main --aumento_datos -b 2048 -tp 30 -tt 10 -or 0.1 -e 500 --label_mode vainilla 
#python -m src.main --aumento_datos -b 2048 -tp 30 -tt 10 -or 0.1 -e 500 --label_mode vainilla 
#python -m src.main --aumento_datos -b 2048 -tp 50 -tt 10 -or 0.1 -e 500 --label_mode vainilla 
#python -m src.main --aumento_datos -b 2048 -tp 50 -tt 10 -or 0.1 -e 500 --label_mode vainilla 
#python -m src.main --aumento_datos -b 2048 -tp 50 -tt 10 -or 0.1 -e 500 --label_mode vainilla 
#python -m src.main --aumento_datos -b 2048 -tp 50 -tt 10 -or 0.1 -e 500 --label_mode vainilla 
#python -m src.main --aumento_datos -b 2048 -tp 80 -tt 16 -or 0.1 -e 500 --label_mode vainilla 
#python -m src.main --aumento_datos -b 2048 -tp 80 -tt 16 -or 0.1 -e 500 --label_mode vainilla 
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
