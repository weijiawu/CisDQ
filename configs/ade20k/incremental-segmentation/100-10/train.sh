#! /bin/bash
ls

wdir="output/ade20k/100-10_cisdq3"

python train_net.py --num-gpus 8 \
--config-file configs/ade20k/incremental-segmentation/100-10/step1.yaml \
OUTPUT_DIR ${wdir}/step1 \
MODEL.WEIGHTS output/ade20k/100-50/step0/cur.pth

python fixckpoint.py \
--i ${wdir}/step1/model_final.pth \
--o ${wdir}/step1/cur.pth

python train_net.py --num-gpus 8 \
--config-file configs/ade20k/incremental-segmentation/100-10/step2.yaml \
OUTPUT_DIR ${wdir}/step2 \
MODEL.WEIGHTS ${wdir}/step1/cur.pth

python fixckpoint.py \
--i ${wdir}/step2/model_final.pth \
--o ${wdir}/step2/cur.pth

python train_net.py --num-gpus 8 \
--config-file configs/ade20k/incremental-segmentation/100-10/step3.yaml \
OUTPUT_DIR ${wdir}/step3 \
MODEL.WEIGHTS ${wdir}/step2/cur.pth

python fixckpoint.py \
--i ${wdir}/step3/model_final.pth \
--o ${wdir}/step3/cur.pth

python train_net.py --num-gpus 8 \
--config-file configs/ade20k/incremental-segmentation/100-10/step4.yaml \
OUTPUT_DIR ${wdir}/step4 \
MODEL.WEIGHTS ${wdir}/step3/cur.pth

python fixckpoint.py \
--i ${wdir}/step4/model_final.pth \
--o ${wdir}/step4/cur.pth

python train_net.py --num-gpus 8 \
--config-file configs/ade20k/incremental-segmentation/100-10/step5.yaml \
OUTPUT_DIR ${wdir}/step5 \
MODEL.WEIGHTS ${wdir}/step4/cur.pth

python fixckpoint.py \
--i ${wdir}/step5/model_final.pth \
--o ${wdir}/step5/cur.pth




