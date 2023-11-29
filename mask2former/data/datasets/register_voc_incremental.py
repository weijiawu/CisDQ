# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import detection_utils as utils
import tqdm
# from detectron2.data.datasets import load_sem_seg
from detectron2.utils.file_io import PathManager
import logging
logger = logging.getLogger(__name__)
import numpy as np
import json




classes = {
    0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'
}

VOC_SEM_SEG = ['background','aeroplane', 'bicycle', 'bird',
               'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor'
]


tasks_voc = {
    "offline": {
        0: list(range(21)),
    },
    "19-1": {
        0: list(range(20)),
        1: [20],
    },
    "15-5": {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        1: [16, 17, 18, 19, 20]
    },
    "15-1":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            1: [16],
            2: [17],
            3: [18],
            4: [19],
            5: [20]
        },
    "10-1":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            1: [11],
            2: [12],
            3: [13],
            4: [14],
            5: [15],
            6: [16],
            7: [17],
            8: [18],
            9: [19],
            10: [20]
        },
    "5-5s":
        {
            0: [0, 1, 2, 3, 4],
            1: [5, 6, 7, 8, 9],
            2: [10, 11, 12, 13, 14],
            3: [15, 16, 17, 18, 19],
        },
    "10-10s":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            1: [10],
            2: [11],
            3: [12],
            4: [13],
            5: [14],
            6: [15],
            7: [16],
            8: [17],
            9: [18],
            10: [19]
        },

    "15-5s_b":
        {
0: [0, 12, 9, 20, 7, 15, 8, 14, 16, 5, 19, 4, 1, 13, 2, 11],
1: [17], 2: [3], 3: [6], 4: [18], 5: [10]
        },
    "15-5s_c":
        {
0: [0, 13, 19, 15, 17, 9, 8, 5, 20, 4, 3, 10, 11, 18, 16, 7],
1: [12], 2: [14], 3: [6], 4: [1], 5: [2]
        },
    "15-5s_d":
        {
0: [0, 15, 3, 2, 12, 14, 18, 20, 16, 11, 1, 19, 8, 10, 7, 17],
1: [6], 2: [5], 3: [13], 4: [9], 5: [4]
        },
    "15-5s_e":
        {
0: [0, 7, 5, 3, 9, 13, 12, 14, 19, 10, 2, 1, 4, 16, 8, 17],
1: [15], 2: [18], 3: [6], 4: [11], 5: [20]
        },
    "15-5s_f":
        {
0: [0, 7, 13, 5, 11, 9, 2, 15, 12, 14, 3, 20, 1, 16, 4, 18],
1: [8], 2: [6], 3: [10], 4: [19], 5: [17]
        },
    "15-5s_g":
        {
0: [0, 7, 5, 9, 1, 15, 18, 14, 3, 20, 10, 4, 19, 11, 17, 16],
1: [12], 2: [8], 3: [6], 4: [2], 5: [13]
        },
    "15-5s_h":
        {
0: [0, 12, 9, 19, 6, 4, 10, 5, 18, 14, 15, 16, 3, 8, 7, 11],
1: [13], 2: [2], 3: [20], 4: [17], 5: [1]
        },
    "15-5s_i":
        {
0: [0, 13, 10, 15, 8, 7, 19, 4, 3, 16, 12, 14, 11, 5, 20, 6],
1: [2], 2: [18], 3: [9], 4: [17], 5: [1]
        },
    "15-5s_j":
        {
0: [0, 1, 14, 9, 5, 2, 15, 8, 20, 6, 16, 18, 7, 11, 10, 19],
1: [3], 2: [4], 3: [17], 4: [12], 5: [13]
        },
    "15-5s_k":
        {
0: [0, 16, 13, 1, 11, 12, 18, 6, 14, 5, 3, 7, 9, 20, 19, 15],
1: [4], 2: [2], 3: [10], 4: [8], 5: [17]
        },
    "15-5s_l":
        {
0: [0, 10, 7, 6, 19, 16, 8, 17, 1, 14, 4, 9, 3, 15, 11, 12],
1: [2], 2: [18], 3: [20], 4: [13], 5: [5]
        },
    "15-5s_m":
        {
0: [0, 18, 4, 14, 17, 12, 10, 7, 3, 9, 1, 8, 15, 6, 13, 2],
1: [5], 2: [11], 3: [20], 4: [16], 5: [19]
        },
    "15-5s_n":
        {
0: [0, 5, 4, 13, 18, 14, 10, 19, 15, 7, 9, 3, 2, 8, 16, 20],
1: [1], 2: [12], 3: [11], 4: [6], 5: [17]
        },
    "15-5s_o":
        {
0: [0, 9, 12, 13, 18, 7, 1, 15, 17, 10, 8, 4, 5, 20, 16, 6],
1: [14], 2: [19], 3: [11], 4: [2], 5: [3]
        },
        #
    "15-5s_p":
        {
0: [0, 9, 12, 13, 18, 2, 11, 15, 17, 10, 8, 4, 5, 20, 16, 6],
1: [14], 2: [19], 3: [1], 4: [7], 5: [3]
        },
    "15-5s_q":
        {
0: [0, 3, 14, 13, 18, 2, 11, 15, 17, 10, 8, 4, 5, 20, 16, 6],
1: [12], 2: [19], 3: [1], 4: [7], 5: [9]
        },
    "15-5s_r":
        {
0: [0, 3, 14, 13, 1, 2, 11, 15, 17, 7, 8, 4, 5, 9, 16, 19],
1: [12], 2: [6], 3: [18], 4: [10], 5: [20]
        },
    "15-5s_s":
        {
0: [0, 3, 14, 6, 1, 2, 11, 12, 17, 7, 20, 4, 5, 9, 16, 19],
1: [15], 2: [13], 3: [18], 4: [10], 5: [8]
        }, #
    "15-5s_t":
        {
0: [0, 3, 15, 13, 1, 2, 11, 18, 17, 7, 20, 8, 5, 9, 16, 19],
1: [14], 2: [6], 3: [12], 4: [10], 5: [4]
        },
    "15-5s_u":
        {
0: [0, 3, 15, 13, 14, 6, 11, 18, 17, 7, 20, 8, 4, 9, 16, 10],
1: [1], 2: [2], 3: [12], 4: [19], 5: [5]
        },
    "15-5s_v":
        {
0: [0, 1, 2, 12, 14, 6, 19, 18, 17, 5, 20, 8, 4, 9, 16, 10],
1: [3], 2: [15], 3: [13], 4: [11], 5: [7]
        },
    "15-5s_w":
        {
0: [0, 1, 2, 12, 14, 13, 19, 18, 7, 11, 20, 8, 4, 9, 16, 10],
1: [3], 2: [15], 3: [6], 4: [5], 5: [17]
        },
}

def _get_voc_full_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing, so all ids are shifted by 1.
    # stuff_ids = [k["id"] for k in ADE20K_SEM_SEG_FULL_CATEGORIES]
    # assert len(stuff_ids) == 847, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    # stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    # stuff_classes = [k["name"] for k in ADE20K_SEM_SEG_FULL_CATEGORIES]

    ret = {
        # "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": VOC_SEM_SEG,
        "tasks_ade": tasks_voc
    }
    return ret

def load_sem_seg(gt_root, image_root, training_classes, gt_ext="png", image_ext="jpg"):
    """
    Load semantic segmentation datasets. All files under "gt_root" with "gt_ext" extension are
    treated as ground truth annotations and all files under "image_root" with "image_ext" extension
    as input images. Ground truth and input images are matched using file paths relative to
    "gt_root" and "image_root" respectively without taking into account file extensions.
    This works for COCO as well as some other datasets.

    Args:
        gt_root (str): full path to ground truth semantic segmentation files. Semantic segmentation
            annotations are stored as images with integer values in pixels that represent
            corresponding semantic labels.
        image_root (str): the directory where the input images are.
        gt_ext (str): file extension for ground truth annotations.
        image_ext (str): file extension for input images.

    Returns:
        list[dict]:
            a list of dicts in detectron2 standard format without instance-level
            annotation.

    Notes:
        1. This function does not read the image and ground truth files.
           The results do not have the "image" and "sem_seg" fields.
    """

    # We match input images with ground truth based on their relative filepaths (without file
    # extensions) starting from 'image_root' and 'gt_root' respectively.
    def file2id(folder_path, file_path):
        # extract relative path starting from `folder_path`
        image_id = os.path.normpath(os.path.relpath(file_path, start=folder_path))
        # remove file extension
        image_id = os.path.splitext(image_id)[0]
        return image_id

    input_files = sorted(
        (os.path.join(image_root, f) for f in PathManager.ls(image_root) if f.endswith(image_ext)),
        key=lambda file_path: file2id(image_root, file_path),
    )
    gt_files = sorted(
        (os.path.join(gt_root, f) for f in PathManager.ls(gt_root) if f.endswith(gt_ext)),
        key=lambda file_path: file2id(gt_root, file_path),
    )

    assert len(gt_files) > 0, "No annotations found in {}.".format(gt_root)

    # Use the intersection, so that val2017_100 annotations can run smoothly with val2017 images
    if len(input_files) != len(gt_files):
        logger.warn(
            "Directory {} and {} has {} and {} files, respectively.".format(
                image_root, gt_root, len(input_files), len(gt_files)
            )
        )
        input_basenames = [os.path.basename(f)[: -len(image_ext)] for f in input_files]
        gt_basenames = [os.path.basename(f)[: -len(gt_ext)] for f in gt_files]
        intersect = list(set(input_basenames) & set(gt_basenames))
        # sort, otherwise each worker may obtain a list[dict] in different order
        intersect = sorted(intersect)
        logger.warn("Will use their intersection of {} files.".format(len(intersect)))
        input_files = [os.path.join(image_root, f + image_ext) for f in intersect]
        gt_files = [os.path.join(gt_root, f + gt_ext) for f in intersect]

    logger.info(
        "Loaded {} images with semantic segmentation from {}".format(len(input_files), image_root)
    )

    if training_classes is not None:
        txt = 'datasets/PascalVOC12/splits/train_aug.txt'
        with open(txt,'r') as fr:
            training_lst = fr.readlines()
        training_images_lst = [os.path.basename(el.split(' ')[0]) for el in training_lst]
        training_gts_lst = [os.path.basename(el.split(' ')[1].strip()) for el in training_lst]
        input_files = [el for el in input_files if os.path.basename(el) in training_images_lst]
        gt_files = [el for el in gt_files if os.path.basename(el) in training_gts_lst]

        dataset_dicts = []
        for (img_path, gt_path) in zip(input_files, gt_files):
            record = {}
            record["file_name"] = img_path
            record["sem_seg_file_name"] = gt_path
            dataset_dicts.append(record)

        if training_classes is not None:
            cachename = os.path.join('cache', str(training_classes[0]) + '_' + str(training_classes[-1]) + '.json')
            if os.path.exists(cachename):
                print(f'load cached anns from {cachename}')
                with open(cachename, 'r') as fr:
                    dataset_dicts = json.load(fr)
            else:
                if 0 in training_classes:
                    training_classes.remove(0) # remove the background
                dataset_dicts = filter_training_data(dataset_dicts, training_classes)
                with open(cachename, 'w') as fw:
                    json.dump(dataset_dicts, fw)
    else:
        txt = 'datasets/PascalVOC12/splits/val.txt'
        with open(txt, 'r') as fr:
            training_lst = fr.readlines()
        training_images_lst = [os.path.basename(el.split(' ')[0]) for el in training_lst]
        training_gts_lst = [os.path.basename(el.split(' ')[1].strip()) for el in training_lst]
        input_files = [el for el in input_files if os.path.basename(el) in training_images_lst]
        gt_files = [el for el in gt_files if os.path.basename(el) in training_gts_lst]

        dataset_dicts = []
        for (img_path, gt_path) in zip(input_files, gt_files):
            record = {}
            record["file_name"] = img_path
            record["sem_seg_file_name"] = gt_path
            dataset_dicts.append(record)

    print(f'filter {len(dataset_dicts)} samples')
    return dataset_dicts

def filter_training_data(dataset_dicts,training_classes=None):
    new_dataset_dicts = []
    for dataset_dict in tqdm.tqdm(dataset_dicts):
        gt_path = dataset_dict['sem_seg_file_name']
        file = utils.read_image(gt_path).astype("long")
        gt_classes = set(np.unique(file).tolist())
        training_classes = set(training_classes)
        if len(training_classes.intersection(gt_classes))>0:
            new_dataset_dicts.append(dataset_dict)
    return new_dataset_dicts

def register_all_voc_full(root):
    root = os.path.join(root, "PascalVOC12")
    meta = _get_voc_full_meta()

    image_dir = os.path.join(root, 'JPEGImages')
    gt_dir = os.path.join(root, 'SegmentationClassAug')

    for task,split in tasks_voc.items():
        nsteps = len(split.values())
        step_list = list(range(nsteps))

        # 为每个step分别注册标注信息
        for step, training_classes in split.items():
            new_name = f"voc_incremental_{task}_{step}"

            DatasetCatalog.register(
                new_name, lambda x=image_dir, y=gt_dir, z=training_classes: load_sem_seg(y, x, z, gt_ext="png", image_ext="jpg")
            )
            MetadataCatalog.get(new_name).set(
                stuff_classes=VOC_SEM_SEG[:],
                training_classes = tasks_voc[task][step],
                task = task,
                step = step,
                image_root=image_dir,
                sem_seg_root=gt_dir,
                evaluator_type="sem_seg",
                ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
            )

def register_val_voc(root):
    root = os.path.join(root, "PascalVOC12")
    image_dir = os.path.join(root, "JPEGImages")
    gt_dir = os.path.join(root, "SegmentationClassAug")
    name = "voc_val"
    DatasetCatalog.register(
        name, lambda x=image_dir, y=gt_dir, z=None: load_sem_seg(y, x, z, gt_ext="png", image_ext="jpg")
    )
    MetadataCatalog.get(name).set(
        stuff_classes=VOC_SEM_SEG[:],
        image_root=image_dir,
        sem_seg_root=gt_dir,
        evaluator_type="sem_seg",
        ignore_label=255,
    )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_voc_full(_root)
register_val_voc(_root)
