#! /bin/bash
ls

wdir="output/ade20k/50-50_cisdq_weijia"

python train_net.py --num-gpus 8 \
--config-file configs/ade20k/incremental-segmentation/50-50/step0.yaml \
OUTPUT_DIR output/ade20k/50-50/step0

#python fixckpoint.py \
#--i output/ade20k/50-50/step0/model_final.pth \
#--o output/ade20k/50-50/step0/cur.pth

python train_net.py --num-gpus 8 \
--config-file configs/ade20k/incremental-segmentation/50-50/step1.yaml \
OUTPUT_DIR ${wdir}/step1 \
MODEL.WEIGHTS output/ade20k/50-50_final/step0/cur.pth

# python fixckpoint.py \
# --i ${wdir}/step1/model_final.pth \
# --o ${wdir}/step1/cur.pth

python train_net.py --num-gpus 8 \
--config-file configs/ade20k/incremental-segmentation/50-50/step2.yaml \
OUTPUT_DIR ${wdir}/step2 \
MODEL.WEIGHTS ${wdir}/step1/cur.pth

# python fixckpoint.py \
# --i ${wdir}/step2/model_final.pth \
# --o ${wdir}/step2/cur.pth





