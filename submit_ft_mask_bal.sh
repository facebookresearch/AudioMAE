#!/bin/bash

if [ -z "$1" ]
then
	blr=1e-3
else
	blr=$1
fi

if [ -z "$2" ]
then
	mask_t_prob=0.0
else
	mask_t_prob=$2
fi

if [ -z "$3" ]
then
	mask_f_prob=0.0
else
	mask_f_prob=$3
fi

if [ -z "$4" ]
then
	ckpt=/checkpoint/berniehuang/experiments/53415548/checkpoint-20.pth
else
	ckpt=$4
fi

audioset_train_json=/checkpoint/berniehuang/ast/egs/audioset/data/datafiles/train.json
audioset_train_all_json=/checkpoint/berniehuang/ast/egs/audioset/data/datafiles/train_all.json
audioset_eval_json=/checkpoint/berniehuang/ast/egs/audioset/data/datafiles/eval_19k.json
audioset_label=/checkpoint/berniehuang/ast/egs/audioset/data/class_labels_indices.csv
audioset_train_weight=/checkpoint/berniehuang/mae/data/audioset/weight_train.csv
audioset_train_all_weight=/checkpoint/berniehuang/mae/data/audioset/weight_train_all.csv
dataset=audioset

python submitit_finetune.py \
    --nodes 8 \
    --model vit_base_patch16 \
    --dataset $dataset \
    --data_train $audioset_train_all_json \
    --data_eval $audioset_eval_json \
    --label_csv $audioset_label \
    --weight_csv $audioset_train_all_weight \
    --finetune $ckpt \
    --blr $blr \
    --dist_eval \
    --batch_size 4 \
    --roll_mag_aug True \
    --mask_t_prob $mask_t_prob \
    --mask_f_prob $mask_f_prob \
    --first_eval_ep 20 \
    --epochs 100 \
    --warmup_epochs 10 \
    --weight_sampler True \
    --replacement False \
    --distributed_wrapper True \
    --mask_2d True \


#audioset_train_all_video_json=/checkpoint/berniehuang/ast/egs/audioset/data/datafiles/train_all_video.json
#audioset_eval_video_json=/checkpoint/berniehuang/ast/egs/audioset/data/datafiles/eval_video.json
#audioset_train_all_video_weight=/checkpoint/berniehuang/mae/data/audioset/weight_train_all_video.csv

