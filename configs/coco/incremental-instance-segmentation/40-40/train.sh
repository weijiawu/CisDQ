#! /bin/bash
ls

wdir="output/cocoins/40-40_cisdq4"

#python train_net.py --num-gpus 8 \
#--config-file configs/coco/incremental-instance-segmentation/40-40/step0.yaml \
#OUTPUT_DIR ${wdir}/step0

#python fixckpoint.py \
#--i ${wdir}/step0/model_final.pth \
#--o ${wdir}/step0/cur.pth

python train_net.py --num-gpus 8 \
--config-file configs/coco/incremental-instance-segmentation/40-40/step1.yaml \
OUTPUT_DIR ${wdir}/step1 \
MODEL.WEIGHTS output/cocoins/40-40_cisdq/step0/cur.pth

python fixckpoint.py \
--i ${wdir}/step1/model_final.pth \
--o ${wdir}/step1/cur.pth


