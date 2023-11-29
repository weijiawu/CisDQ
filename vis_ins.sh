#! /bin/bash
ls

image='000000194716'

model='cisdq'
weight='output/cocoins/40-40_'${model}
output='demo/'${model}'/'${image}
image='datasets/coco/val2017/'${image}'.jpg'

mkdir -p output

for i in {0,1}
do
    echo ${i}
    python demo/demo.py \
    --config-file configs/coco/incremental-instance-segmentation/40-40/step${i}.yaml \
    --input ${image} \
    --output ${output}/step${i}.jpg \
    --opts MODEL.WEIGHTS ${weight}/step${i}/cur.pth
done

#model='plop'
#weight='output/ade20k/100-5_plop'
#output='demo/plop/'${image}
#
#mkdir -p output
#
#for i in {0,1,2,3,4,5,6,7,8,9,10}
#do
#    echo ${i}
#    python demo/demo.py \
#    --config-file configs/ade20k/incremental-segmentation/100-5/step${i}.yaml \
#    --input ${image} \
#    --output ${output}/step${i}.npy \
#    --opts MODEL.WEIGHTS ${weight}/step${i}/cur.pth
#done
