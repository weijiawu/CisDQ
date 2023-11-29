# #! /bin/bash
# ls

# image='ADE_val_00000322'

# model='plop'
# weight='output/ade20k/100-5_'${model}
# output='demo/'${model}'/'${image}
# image='datasets/ADEChallengeData2016/images/validation/'${image}'.jpg'

# mkdir -p output

# for i in {0,1,2,3,4,5,6,7,8,9,10}
# do
#     echo ${i}
#     python demo/demo.py \
#     --config-file configs/ade20k/incremental-segmentation/100-5/step${i}.yaml \
#     --input ${image} \
#     --output ${output}/step${i}.npy \
#     --opts MODEL.WEIGHTS ${weight}/step${i}/cur.pth
# done

#! /bin/bash
ls

# image='ADE_val_00000014'
# # /output/ade20k/100-10_cisdq/step0/cur.pth
# model='cisdq'
# weight='output/ade20k/100-10_'${model}
# output='demo/'${model}'/'
# image='datasets/ADEChallengeData2016/images/validation/'${image}'.jpg'

# mkdir -p output

# python demo/demo.py \
#     --config-file configs/ade20k/incremental-segmentation/100-10/step0.yaml \
#     --input ${image} \
#     --output ${output}/step0.jpg \
#     --opts MODEL.WEIGHTS ${weight}/step5/cur_fix.pth
 
image='ADE_val_00000014'
# /output/ade20k/100-10_cisdq/step0/cur.pth  15-1_cisdq
model='cisdq'
weight='output/voc/15-1_'${model}
output='demo/'${model}'/'
image='datasets/ADEChallengeData2016/images/validation/'${image}'.jpg'

mkdir -p output

python demo/demo.py \
    --config-file configs/voc/incremental-segmentation/15-1/step5.yaml \
    --input ${image} \
    --output ${output}/step0.jpg \
    --opts MODEL.WEIGHTS ${weight}/step5/cur_fix.pth
