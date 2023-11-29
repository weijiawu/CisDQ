#! /bin/bash
ls


python train_net.py --num-gpus 8 \
--config-file configs/ade20k/incremental-segmentation/100-5/step9.yaml \
OUTPUT_DIR output/ade20k/100-5/step9 \
MODEL.WEIGHTS output/ade20k/100-5/step8/cur.pth

python fixckpoint.py \
--i output/ade20k/100-5/step9/model_final.pth \
--o output/ade20k/100-5/step9/cur.pth

python train_net.py --num-gpus 8 \
--config-file configs/ade20k/incremental-segmentation/100-5/step10.yaml \
OUTPUT_DIR output/ade20k/100-5/step10 \
MODEL.WEIGHTS output/ade20k/100-5/step9/cur.pth

python fixckpoint.py \
--i output/ade20k/100-5/step10/model_final.pth \
--o output/ade20k/100-5/step10/cur.pth




