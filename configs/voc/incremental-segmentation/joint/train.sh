#! /bin/bash
ls

python train_net.py --num-gpus 8 \
--config-file configs/voc/incremental-segmentation/joint/step0.yaml \
OUTPUT_DIR output/voc/joint/step0
#OUTPUT_DIR output/debug

python fixckpoint.py \
--i output/voc/joint/step0/model_final.pth \
--o output/voc/joint/step0/cur.pth


