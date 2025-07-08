#!/usr/bin/env bash

CKPT_FILE="./checkpoints/blendedmvs/model_000017.ckpt"
# CKPT_FILE="./checkpoints/blendedmvs/model_000015.ckpt"

ETH3D_TESTING="/mnt/ssd/dataset/eth3d/eth3d_high_res_test"

OUT_DIR="/mnt/ssd/dataset/miper_student_eval/miper_kd_output_tiaocan"

#python eval.py --dataset=eth3d --split train --batch_size=1 --n_views 7 --iteration 4 \
#--testpath=$ETH3D_TESTING --geo_pixel_thres=1 --geo_depth_thres=0.01 --photo_thres=0.3 \
#--outdir=$OUT_DIR --loadckpt $CKPT_FILE $@

python eval_gpu.py --dataset=eth3d --split train --batch_size=1 --n_views 7 --iteration 4 \
--testpath=$ETH3D_TESTING --geo_pixel_thres=1 --geo_depth_thres=0.01 --photo_thres=0.1 \
--outdir=$OUT_DIR --loadckpt $CKPT_FILE $@




