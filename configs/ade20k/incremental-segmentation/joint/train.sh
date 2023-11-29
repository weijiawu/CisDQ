#! /bin/bash
ls

python train_net.py --num-gpus 8 \
--config-file configs/ade20k/incremental-segmentation/joint/step0.yaml \
OUTPUT_DIR output/ade20k/joint/step0

python fixckpoint.py \
--i output/ade20k/joint/step0/model_final.pth \
--o output/ade20k/joint/step0/cur.pth


