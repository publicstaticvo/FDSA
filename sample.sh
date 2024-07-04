#! /bin/bash

datadir=~/datasets/semeval/
model=../bertmodel
ml=150

python main.py --data_dir $datadir --max_length $ml --model_path $model
