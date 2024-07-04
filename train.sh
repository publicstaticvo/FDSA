#! /bin/bash

model=./bertmodel/

python main.py --model_path $model --beta 1e-3 --data_dir ./data/semeval/ --task_name semeval --max_length 150 --seed 1
python main.py --model_path $model --train_batch_size 10 --beta 0.1 --data_dir ./data/kbp37/ --task_name kbp37 --max_length 190

