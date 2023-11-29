#! /bin/bash
ls

wdir="output/ade20kins/25-25_cisdq"

python train_net.py --num-gpus 8 \
--config-file configs/ade20k/incremental-instance-segmentation/25-25/step0.yaml \
OUTPUT_DIR ${wdir}/step0

python fixckpoint.py \
--i ${wdir}/step0/model_final.pth \
--o ${wdir}/step0/cur.pth

python train_net.py --num-gpus 8 \
--config-file configs/ade20k/incremental-instance-segmentation/25-25/step1.yaml \
OUTPUT_DIR ${wdir}/step1 \
MODEL.WEIGHTS ${wdir}/step0/cur.pth

python fixckpoint.py \
--i ${wdir}/step1/model_final.pth \
--o ${wdir}/step1/cur.pth

python train_net.py --num-gpus 8 \
--config-file configs/ade20k/incremental-instance-segmentation/25-25/step2.yaml \
OUTPUT_DIR ${wdir}/step2 \
MODEL.WEIGHTS ${wdir}/step1/cur.pth

python fixckpoint.py \
--i ${wdir}/step2/model_final.pth \
--o ${wdir}/step2/cur.pth

python train_net.py --num-gpus 8 \
--config-file configs/ade20k/incremental-instance-segmentation/25-25/step3.yaml \
OUTPUT_DIR ${wdir}/step3 \
MODEL.WEIGHTS ${wdir}/step2/cur.pth

python fixckpoint.py \
--i ${wdir}/step3/model_final.pth \
--o ${wdir}/step3/cur.pth
