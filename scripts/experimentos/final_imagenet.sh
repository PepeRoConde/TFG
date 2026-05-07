python -m src.main -a CRATE_tiny -e 3 -j 2 --label_smoothing 0.1 -b 5 -lr 2.4e-4 --wd 0.5 --dataset imagenet --warmup_steps 5 --use_cosine_scheduler -tp 224 -tt 8 -runs_dir final_imagenet

python -m src.main -a CRATE_tiny -e 3 -j 3 --label_smoothing 0.1 -b 32 -lr 2.4e-4 --wd 0.5 --dataset imagenet --warmup_steps 5 --use_cosine_scheduler -tp 224 -tt 8 -runs_dir final_imagene --use_amp
