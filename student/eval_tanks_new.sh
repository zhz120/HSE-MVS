#!/usr/bin/env bash

CKPT_FILE="./checkpoints/blendedmvs/model_000018.ckpt"
# CKPT_FILE="./checkpoints/blendedmvs/model_000015.ckpt"

TANK_TESTING="/home/god/mvs/data/TankandTemples/"

OUT_DIR="./outputs_tanks_blended_iter8/"

python eval_gpu.py --dataset=tanks --split intermediate --batch_size=1 --n_views 7 --iteration 8 \
--testpath=$TANK_TESTING --geo_pixel_thres=1 --geo_depth_thres=0.01 --photo_thres=0.3 \
--outdir=$OUT_DIR --loadckpt $CKPT_FILE $@

#python eval.py --dataset=tanks --split advanced --batch_size=1 --n_views 7 --iteration 4 \
#--testpath=$TANK_TESTING --geo_pixel_thres=1 --geo_depth_thres=0.01 --photo_thres=0.3 \
#--outdir=$OUT_DIR --loadckpt $CKPT_FILE $@













#CKPT_FILE="./checkpoints/dtu/model_000019.ckpt"
## CKPT_FILE="./checkpoints/blendedmvs/model_000015.ckpt"
#
#CUSTOM_TESTING="/remote-home/zhaohaoliang/zhz/data/custom/wall/dense/"
#
#OUT_DIR="/remote-home/zhaohaoliang/zhz/IterMVS-main/outputs_wall/"
#
#python eval_gpu.py --dataset=custom --batch_size=1 --n_views 7 --iteration 4 --img_wh 2560 1920 \
#--testpath=$CUSTOM_TESTING --geo_pixel_thres=1 --geo_depth_thres=0.01 --photo_thres=0.15 \
#--outdir=$OUT_DIR --loadckpt $CKPT_FILE $@