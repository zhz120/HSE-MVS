#!/usr/bin/env bash

# train on BlendedMVS
MVS_TRAINING="/mnt/ssd/dataset/blendedmvs"

LOG_DIR="./checkpoints/blendedmvs"

#python train.py --dataset blendedmvs --batch_size 2 --epochs 1 --lr 0.001 --lrepochs 4,8,12:2 \
#--iteration 4 \
#--trainpath=$MVS_TRAINING1 --trainlist lists/blendedmvs/train.txt --vallist lists/blendedmvs/val.txt \
#--logdir=$LOG_DIR $@

python train.py --dataset blendedmvs --batch_size 2 --epochs 20 --lr 0.001 --lrepochs 4,8,12,16,20:2 --check_depth_path /home/1Tm2/zhz/KD-MVS/KD-MVS/kdmvs_blended/outputs/00_teacher_model_kd_iter1/checked_depth_normal_version --regress \
--iteration 4 \
--trainpath=$MVS_TRAINING --trainlist lists/blendedmvs/train.txt --vallist lists/blendedmvs/val.txt \
--logdir=$LOG_DIR $@
