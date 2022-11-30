#!/bin/bash
if [ -z "$1" ]
then
    blr=2e-4
else
    blr=$1
fi




audioset_train_all_video_json=/checkpoint/berniehuang/ast/egs/audioset/data/datafiles/train_all_video.json
audioset_train_all_json=/checkpoint/berniehuang/ast/egs/audioset/data/datafiles/train_all.json
audioset_label=/checkpoint/berniehuang/ast/egs/audioset/data/class_labels_indices.csv


dataset=audioset

python submitit_pretrain.py \
--use_volta32 \
--nodes 8 \
--batch_size 8 \
--norm_pix_loss True \
--model mae_vit_base_patch16 \
--mask_ratio 0.8 \
--epochs 33 \
--warmup_epochs 3 \
--save_every_epoch 4 \
--blr $blr --weight_decay 0.0001 \
--dataset $dataset \
--data_train $audioset_train_all_json \
--label_csv $audioset_label \
--roll_mag_aug True \
--decoder_mode 1 \



