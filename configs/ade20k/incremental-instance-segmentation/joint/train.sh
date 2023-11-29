#! /bin/bash
ls

wdir="output/ade20kins/joint"

python train_net.py --num-gpus 2 \
--config-file configs/ade20k/incremental-instance-segmentation/joint/step0.yaml \
OUTPUT_DIR ${wdir}/step0

python fixckpoint.py \
--i ${wdir}/step0/model_final.pth \
--o ${wdir}/step0/cur.pth


