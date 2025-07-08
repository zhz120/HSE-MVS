#!/usr/bin/env bash
# run this script in the root path of KD-MVS
TESTPATH="/mnt/ssd/dataset/dtu_test_kdmvs/dtu" # path to dataset dtu_test
TESTLIST="lists/dtu/test.txt"
#CKPT_FILE="/home/1Tm2/zhz/KD-MVS/KD-MVS/KD-MVS-master1/outputs/00_te_student_kd_0_1_10_train_iter3_normal/model_000010.ckpt" # path to checkpoint file
CKPT_FILE="/home/1Tm2/zhz/KD-MVS/KD-MVS/KD-MVS-master1/ckpt/model_kd_dtu.ckpt" # path to checkpoint file
FUSIBILE_EXE="/home/1Tm2/zhz/fusibile/fusibile-master/fusibile" # path to gipuma fusible file
OUTDIR="/mnt/ssd/dataset/check_blended_all/kd_origin_test/kdmvs_data_test_dtu_te0_kd0_1_10_10_iter3_torch1.13" # path to save outputs
if [ ! -d $OUTDIR ]; then
	mkdir -p $OUTDIR
fi

python test.py \
--dataset=general_eval \
--batch_size=1 \
--testpath=$TESTPATH  \
--testlist=$TESTLIST \
--loadckpt=$CKPT_FILE \
--outdir=$OUTDIR \
--fusibile_exe_path=$FUSIBILE_EXE \
--interval_scale=1.06