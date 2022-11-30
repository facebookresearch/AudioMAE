#!/bin/bash

if [ -z "$1" ]
then
	blr=1e-3
else
	blr=$1
fi

if [ -z "$2" ]
then
	ckpt=/checkpoint/berniehuang/experiments/53415548/checkpoint-20.pth
else
	ckpt=$2
fi

if [ -z "$3" ]
then
	model=vit_base_patch16
else
	model=$3
fi

audioset_train_json=/checkpoint/berniehuang/ast/egs/audioset/data/datafiles/train.json
audioset_train_all_json=/checkpoint/berniehuang/ast/egs/audioset/data/datafiles/train_all.json
audioset_eval_json=/checkpoint/berniehuang/ast/egs/audioset/data/datafiles/eval.json
audioset_label=/checkpoint/berniehuang/ast/egs/audioset/data/class_labels_indices.csv
dataset=audioset

python submitit_finetune.py \
    --nodes 8 \
    --model $model \
    --dataset $dataset \
    --data_train $audioset_train_all_json \
    --data_eval $audioset_eval_json \
    --label_csv $audioset_label \
    --finetune $ckpt \
    --use_volta32 \
    --blr $blr \
    --epochs 30 \
    --warmup_epochs 2 \
    --first_eval_ep 2 \
    --dist_eval \
    --batch_size 8 \
    --roll_mag_aug True \
