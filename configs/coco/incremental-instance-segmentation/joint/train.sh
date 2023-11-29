#! /bin/bash
ls

wdir="output/cocoins/joint"

python train_net.py --num-gpus 8 \
--config-file configs/coco/incremental-instance-segmentation/joint/step0.yaml \
OUTPUT_DIR ${wdir}/step0

python fixckpoint.py \
--i ${wdir}/step0/model_final.pth \
--o ${wdir}/step0/cur.pth


