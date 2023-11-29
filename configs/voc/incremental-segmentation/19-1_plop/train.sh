#! /bin/bash
ls

wdir="output/voc/19-1_mib"

#python train_net.py --num-gpus 8 \
#--config-file configs/voc/incremental-segmentation/19-1/step0.yaml \
#OUTPUT_DIR ${wdir}/step0

#python fixckpoint.py \
#--i ${wdir}/step0/model_final.pth \
#--o ${wdir}/step0/cur.pth

python train_net.py --num-gpus 8 \
--config-file configs/voc/incremental-segmentation/19-1/step1.yaml \
OUTPUT_DIR ${wdir}/step1 \
MODEL.WEIGHTS ${wdir}/step0/cur.pth

python fixckpoint.py \
--i ${wdir}/step1/model_final.pth \
--o ${wdir}/step1/cur.pth

