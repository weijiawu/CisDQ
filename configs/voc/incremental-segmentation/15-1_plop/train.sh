#! /bin/bash
ls

wdir="output/voc/15-1_plop"

#python train_net.py --num-gpus 2 \
#--config-file configs/voc/incremental-segmentation/joint/step0.yaml \
#OUTPUT_DIR output/voc/joint/step0

#python fixckpoint.py \
#--i output/voc/joint/step0/model_final.pth \
#--o output/voc/joint/step0/cur.pth

python train_net.py --num-gpus 8 \
--config-file configs/voc/incremental-segmentation/15-1/step1.yaml \
OUTPUT_DIR ${wdir}/step1 \
MODEL.WEIGHTS output/voc/15-5_pod_mm/step0/cur.pth

python fixckpoint.py \
--i ${wdir}/step1/model_final.pth \
--o ${wdir}/step1/cur.pth

python train_net.py --num-gpus 8 \
--config-file configs/voc/incremental-segmentation/15-1/step2.yaml \
OUTPUT_DIR ${wdir}/step2 \
MODEL.WEIGHTS ${wdir}/step1/cur.pth

python fixckpoint.py \
--i ${wdir}/step2/model_final.pth \
--o ${wdir}/step2/cur.pth

python train_net.py --num-gpus 8 \
--config-file configs/voc/incremental-segmentation/15-1/step3.yaml \
OUTPUT_DIR ${wdir}/step3 \
MODEL.WEIGHTS ${wdir}/step2/cur.pth

python fixckpoint.py \
--i ${wdir}/step3/model_final.pth \
--o ${wdir}/step3/cur.pth

python train_net.py --num-gpus 8 \
--config-file configs/voc/incremental-segmentation/15-1/step4.yaml \
OUTPUT_DIR ${wdir}/step4 \
MODEL.WEIGHTS ${wdir}/step3/cur.pth

python fixckpoint.py \
--i ${wdir}/step4/model_final.pth \
--o ${wdir}/step4/cur.pth

python train_net.py --num-gpus 8 \
--config-file configs/voc/incremental-segmentation/15-1/step5.yaml \
OUTPUT_DIR ${wdir}/step5 \
MODEL.WEIGHTS ${wdir}/step4/cur.pth

python fixckpoint.py \
--i ${wdir}/step5/model_final.pth \
--o ${wdir}/step5/cur.pth
