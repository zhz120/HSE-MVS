#!/usr/bin/env bash

CKPT_FILE="./checkpoints/blendedmvs/model_000018.ckpt"
# CKPT_FILE="./checkpoints/blendedmvs/model_000015.ckpt"

TANK_TESTING="/home/god/mvs/data/TankandTemples/"

OUT_DIR="./outputs_tanks"

#python3 eval_d2hc.py --dataset=tanks --split intermediate --batch_size=1 --n_views 7 --iteration 4 \
#--testpath=$TANK_TESTING --geo_pixel_thres=1 --geo_depth_thres=0.01 --photo_thres=0.3 \
#--outdir=$OUT_DIR --loadckpt $CKPT_FILE $@

python eval.py --dataset=tanks --split advanced --batch_size=1 --n_views 7 --iteration 4 \
--testpath=$TANK_TESTING --geo_pixel_thres=1 --geo_depth_thres=0.01 --photo_thres=0.3 \
--outdir=$OUT_DIR --loadckpt $CKPT_FILE $@