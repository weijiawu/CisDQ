#! /bin/bash
ls

wdir='output/ade20k/100-50_plop_weijia'

python train_net.py --num-gpus 8 \
--config-file configs/ade20k/incremental-segmentation/100-50/step0.yaml \
OUTPUT_DIR ${wdir}/step0

# python fixckpoint.py \
# --i ${wdir}/step0/model_final.pth \
# --o ${wdir}/step0/cur.pth

python train_net.py --num-gpus 8 \
--config-file configs/ade20k/incremental-segmentation/100-50/step1.yaml \
OUTPUT_DIR ${wdir}/step1 \
MODEL.WEIGHTS output/ade20k/100-50/step0/cur.pth

# python fixckpoint.py \
# --i ${wdir}/step1/model_final.pth \
# --o ${wdir}/step1/cur.pth
