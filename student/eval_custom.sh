#!/usr/bin/env bash

CKPT_FILE="./checkpoints/blendedmvs/model_000018.ckpt"
# CKPT_FILE="./checkpoints/blendedmvs/model_000015.ckpt"

CUSTOM_TESTING="/media/god/usb/zhz/dataset/colmap/guanzi/COLMAP/dense"

OUT_DIR="./outputs_custom_guanzi"

python eval_gpu.py --dataset=custom --batch_size=1 --n_views 7 --iteration 4 --img_wh 960 768 \
--testpath=$CUSTOM_TESTING --geo_pixel_thres=1 --geo_depth_thres=0.01 --photo_thres=0.35 \
--outdir=$OUT_DIR --loadckpt $CKPT_FILE $@
