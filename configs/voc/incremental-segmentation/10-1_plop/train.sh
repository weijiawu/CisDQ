#! /bin/bash
ls

wdir="output/voc/10-1_plop"

#python train_net.py --num-gpus 8 \
#--config-file configs/voc/incremental-segmentation/10-1/step0.yaml \
#OUTPUT_DIR ${wdir}/step0

#python fixckpoint.py \
#--i ${wdir}/step0/model_final.pth \
#--o ${wdir}/step0/cur.pth

python train_net.py --num-gpus 8 \
--config-file configs/voc/incremental-segmentation/10-1/step1.yaml \
OUTPUT_DIR ${wdir}/step1 \
MODEL.WEIGHTS ${wdir}/step0/cur.pth

python fixckpoint.py \
--i ${wdir}/step1/model_final.pth \
--o ${wdir}/step1/cur.pth

python train_net.py --num-gpus 8 \
--config-file configs/voc/incremental-segmentation/10-1/step2.yaml \
OUTPUT_DIR ${wdir}/step2 \
MODEL.WEIGHTS ${wdir}/step1/cur.pth

python fixckpoint.py \
--i ${wdir}/step2/model_final.pth \
--o ${wdir}/step2/cur.pth

python train_net.py --num-gpus 8 \
--config-file configs/voc/incremental-segmentation/10-1/step3.yaml \
OUTPUT_DIR ${wdir}/step3 \
MODEL.WEIGHTS ${wdir}/step2/cur.pth

python fixckpoint.py \
--i ${wdir}/step3/model_final.pth \
--o ${wdir}/step3/cur.pth

python train_net.py --num-gpus 8 \
--config-file configs/voc/incremental-segmentation/10-1/step4.yaml \
OUTPUT_DIR ${wdir}/step4 \
MODEL.WEIGHTS ${wdir}/step3/cur.pth

python fixckpoint.py \
--i ${wdir}/step4/model_final.pth \
--o ${wdir}/step4/cur.pth

python train_net.py --num-gpus 8 \
--config-file configs/voc/incremental-segmentation/10-1/step5.yaml \
OUTPUT_DIR ${wdir}/step5 \
MODEL.WEIGHTS ${wdir}/step4/cur.pth

python fixckpoint.py \
--i ${wdir}/step5/model_final.pth \
--o ${wdir}/step5/cur.pth

python train_net.py --num-gpus 8 \
--config-file configs/voc/incremental-segmentation/10-1/step6.yaml \
OUTPUT_DIR ${wdir}/step6 \
MODEL.WEIGHTS ${wdir}/step5/cur.pth

python fixckpoint.py \
--i ${wdir}/step6/model_final.pth \
--o ${wdir}/step6/cur.pth

python train_net.py --num-gpus 8 \
--config-file configs/voc/incremental-segmentation/10-1/step7.yaml \
OUTPUT_DIR ${wdir}/step7 \
MODEL.WEIGHTS ${wdir}/step6/cur.pth

python fixckpoint.py \
--i ${wdir}/step7/model_final.pth \
--o ${wdir}/step7/cur.pth

python train_net.py --num-gpus 8 \
--config-file configs/voc/incremental-segmentation/10-1/step8.yaml \
OUTPUT_DIR ${wdir}/step8 \
MODEL.WEIGHTS ${wdir}/step7/cur.pth

python fixckpoint.py \
--i ${wdir}/step8/model_final.pth \
--o ${wdir}/step8/cur.pth

python train_net.py --num-gpus 8 \
--config-file configs/voc/incremental-segmentation/10-1/step9.yaml \
OUTPUT_DIR ${wdir}/step9 \
MODEL.WEIGHTS ${wdir}/step8/cur.pth

python fixckpoint.py \
--i ${wdir}/step9/model_final.pth \
--o ${wdir}/step9/cur.pth

python train_net.py --num-gpus 8 \
--config-file configs/voc/incremental-segmentation/10-1/step10.yaml \
OUTPUT_DIR ${wdir}/step10 \
MODEL.WEIGHTS ${wdir}/step9/cur.pth

python fixckpoint.py \
--i ${wdir}/step10/model_final.pth \
--o ${wdir}/step10/cur.pth
