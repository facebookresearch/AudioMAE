#!/bin/bash
#SBATCH --job-name=aud-ft
#SBATCH --partition=learnfair
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
#SBATCH --mem=480GB
#SBATCH --signal=USR1@120
#SBATCH --output=/checkpoint/%u/jobs/%A.out
#SBATCH --error=/checkpoint/%u/jobs/%A.err

audioset_train_json=/checkpoint/berniehuang/ast/egs/audioset/data/datafiles/train.json
audioset_train_all_json=/checkpoint/berniehuang/ast/egs/audioset/data/datafiles/train_all.json
audioset_eval_json=/checkpoint/berniehuang/ast/egs/audioset/data/datafiles/eval_19k.json
audioset_label=/checkpoint/berniehuang/ast/egs/audioset/data/class_labels_indices.csv
dataset=audioset

if [ -z "$1" ]
then
    ckpt='/checkpoint/berniehuang/experiments/419909/checkpoint-99.pth'
else
    ckpt=$1
fi


CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 main_finetune_as.py \
--log_dir /checkpoint/berniehuang/mae/as_exp/$SLURM_JOB_ID \
--output_dir /checkpoint/berniehuang/mae/as_exp/$SLURM_JOB_ID \
--model vit_base_patch16 \
--dataset $dataset \
--data_train $audioset_train_json \
--data_eval $audioset_eval_json \
--label_csv $audioset_label \
--finetune $ckpt \
--batch_size 16 \
--eval \



