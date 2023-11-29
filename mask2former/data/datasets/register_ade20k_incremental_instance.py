# Copyright (c) Facebook, Inc. and its affiliates.
import json
import logging
import numpy as np
import os
from PIL import Image
import tqdm
import contextlib
import datetime
import io
import json
import logging
import os
import shutil
import pycocotools.mask as mask_util
from fvcore.common.timer import Timer
from iopath.common.file_io import file_lock
from PIL import Image

from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
from detectron2.utils.file_io import PathManager


"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""


logger = logging.getLogger(__name__)

from detectron2.data import DatasetCatalog, MetadataCatalog
# from detectron2.data.datasets.coco import load_coco_json, register_coco_instances
from detectron2.utils.file_io import PathManager

ADE_CATEGORIES = [{'id': 7, 'name': 'bed'}, {'id': 8, 'name': 'windowpane'}, {'id': 10, 'name': 'cabinet'}, {'id': 12, 'name': 'person'}, {'id': 14, 'name': 'door'}, {'id': 15, 'name': 'table'}, {'id': 18, 'name': 'curtain'}, {'id': 19, 'name': 'chair'}, {'id': 20, 'name': 'car'}, {'id': 22, 'name': 'painting'}, {'id': 23, 'name': 'sofa'}, {'id': 24, 'name': 'shelf'}, {'id': 27, 'name': 'mirror'}, {'id': 30, 'name': 'armchair'}, {'id': 31, 'name': 'seat'}, {'id': 32, 'name': 'fence'}, {'id': 33, 'name': 'desk'}, {'id': 35, 'name': 'wardrobe'}, {'id': 36, 'name': 'lamp'}, {'id': 37, 'name': 'bathtub'}, {'id': 38, 'name': 'railing'}, {'id': 39, 'name': 'cushion'}, {'id': 41, 'name': 'box'}, {'id': 42, 'name': 'column'}, {'id': 43, 'name': 'signboard'}, {'id': 44, 'name': 'chest of drawers'}, {'id': 45, 'name': 'counter'}, {'id': 47, 'name': 'sink'}, {'id': 49, 'name': 'fireplace'}, {'id': 50, 'name': 'refrigerator'}, {'id': 53, 'name': 'stairs'}, {'id': 55, 'name': 'case'}, {'id': 56, 'name': 'pool table'}, {'id': 57, 'name': 'pillow'}, {'id': 58, 'name': 'screen door'}, {'id': 62, 'name': 'bookcase'}, {'id': 64, 'name': 'coffee table'}, {'id': 65, 'name': 'toilet'}, {'id': 66, 'name': 'flower'}, {'id': 67, 'name': 'book'}, {'id': 69, 'name': 'bench'}, {'id': 70, 'name': 'countertop'}, {'id': 71, 'name': 'stove'}, {'id': 72, 'name': 'palm'}, {'id': 73, 'name': 'kitchen island'}, {'id': 74, 'name': 'computer'}, {'id': 75, 'name': 'swivel chair'}, {'id': 76, 'name': 'boat'}, {'id': 78, 'name': 'arcade machine'}, {'id': 80, 'name': 'bus'}, {'id': 81, 'name': 'towel'}, {'id': 82, 'name': 'light'}, {'id': 83, 'name': 'truck'}, {'id': 85, 'name': 'chandelier'}, {'id': 86, 'name': 'awning'}, {'id': 87, 'name': 'streetlight'}, {'id': 88, 'name': 'booth'}, {'id': 89, 'name': 'television receiver'}, {'id': 90, 'name': 'airplane'}, {'id': 92, 'name': 'apparel'}, {'id': 93, 'name': 'pole'}, {'id': 95, 'name': 'bannister'}, {'id': 97, 'name': 'ottoman'}, {'id': 98, 'name': 'bottle'}, {'id': 102, 'name': 'van'}, {'id': 103, 'name': 'ship'}, {'id': 104, 'name': 'fountain'}, {'id': 107, 'name': 'washer'}, {'id': 108, 'name': 'plaything'}, {'id': 110, 'name': 'stool'}, {'id': 111, 'name': 'barrel'}, {'id': 112, 'name': 'basket'}, {'id': 115, 'name': 'bag'}, {'id': 116, 'name': 'minibike'}, {'id': 118, 'name': 'oven'}, {'id': 119, 'name': 'ball'}, {'id': 120, 'name': 'food'}, {'id': 121, 'name': 'step'}, {'id': 123, 'name': 'trade name'}, {'id': 124, 'name': 'microwave'}, {'id': 125, 'name': 'pot'}, {'id': 126, 'name': 'animal'}, {'id': 127, 'name': 'bicycle'}, {'id': 129, 'name': 'dishwasher'}, {'id': 130, 'name': 'screen'}, {'id': 132, 'name': 'sculpture'}, {'id': 133, 'name': 'hood'}, {'id': 134, 'name': 'sconce'}, {'id': 135, 'name': 'vase'}, {'id': 136, 'name': 'traffic light'}, {'id': 137, 'name': 'tray'}, {'id': 138, 'name': 'ashcan'}, {'id': 139, 'name': 'fan'}, {'id': 142, 'name': 'plate'}, {'id': 143, 'name': 'monitor'}, {'id': 144, 'name': 'bulletin board'}, {'id': 146, 'name': 'radiator'}, {'id': 147, 'name': 'glass'}, {'id': 148, 'name': 'clock'}, {'id': 149, 'name': 'flag'}]

# ids = [12,20,19,82,8,22,15,10,36,43,14,67,39,18,87,41,98,125,7,24,66,30,32,57,135,93,42,134,31,23,142,147,27,47,38,33,69,72,53,81,112,75,108,115,136,110,86,64,148,138,149,85,102,95,76,137,119,89,83,44,116,120,143,127,123,71,132,74,62,49,80,139,92,45,50,35,65,37,126,97,70,124,111,121,90,144,133,118,129,56,55,78,146,88,107,58,73,130,104,103]
ids = [7,8,10,12,14,15,18,19,20,22,23,24,27,30,31,32,33,35,36,37,38,39,41,42,43,44,45,47,49,50,53,55,56,57,58,62,64,65,66,67,69,70,71,72,73,74,75,76,78,80,81,82,83,85,86,87,88,89,90,92,93,95,97,98,102,103,104,107,108,110,111,112,115,116,118,119,120,121,123,124,125,126,127,129,130,132,133,134,135,136,137,138,139,142,143,144,146,147,148,149]

tasks_ade_ins = {
    "offline": {
        0: [ids[x] for x in range(100)]
    },
    "25-25": {
        0: [ids[x] for x in range(0, 25)],
        1: [ids[x] for x in range(25, 50)],
        2: [ids[x] for x in range(50, 75)],
        3: [ids[x] for x in range(75, 100)]
    },
    "50-50": {
        0: [ids[x] for x in range(0, 50)],
        1: [ids[x] for x in range(50, 100)]
    },
    "50-10":
        {
            0: [ids[x] for x in range(0, 50)],
            1: [ids[x] for x in range(50, 60)],
            2: [ids[x] for x in range(60, 70)],
            3: [ids[x] for x in range(70, 80)],
            4: [ids[x] for x in range(80, 90)],
            5: [ids[x] for x in range(90, 100)]
        },
    "50-5":
        {
            0: [ids[x] for x in range(0, 50)],
            1: [ids[x] for x in range(50, 55)],
            2: [ids[x] for x in range(55, 60)],
            3: [ids[x] for x in range(60, 65)],
            4: [ids[x] for x in range(65, 70)],
            5: [ids[x] for x in range(70, 75)],
            6: [ids[x] for x in range(75, 80)],
            7: [ids[x] for x in range(80, 85)],
            8: [ids[x] for x in range(85, 90)],
            9: [ids[x] for x in range(90, 95)],
            10: [ids[x] for x in range(95, 100)],
        }
}


def make_incremental_anns():
    splits = dict()

    imgs_dir = "datasets/ADEChallengeData2016/images/training"
    anns_path = 'datasets/ADEChallengeData2016/ade20k_instance_train.json'
    tasks_dir = 'datasets/ADEChallengeData2016/ade20k_instance_incremental'
    if not os.path.exists(tasks_dir):
        os.mkdir(tasks_dir)
    with open(anns_path, 'r') as fr:
        file = json.load(fr)
    imgs = file['images']
    cats = file['categories']
    anns = file['annotations']
    for task in tasks_ade_ins:
        steps = tasks_ade_ins[task]
        task_dir = os.path.join(tasks_dir,task)
        if not os.path.exists(task_dir):
            os.mkdir(task_dir)
        for step in steps:
            step_path = os.path.join(task_dir,str(step)+'.json')
            if not os.path.exists(step_path):
                training_classes = steps[step]
                step_file = dict()
                step_cats = [el for el in cats if el['id'] in training_classes]
                step_anns = [el for el in anns if el['category_id'] in training_classes]
                step_imgs = set([el['image_id'] for el in step_anns])
                step_imgs = [el for el in imgs if el['id'] in step_imgs]
                step_file["images"] = step_imgs
                step_file["categories"] = step_cats
                step_file["annotations"] = step_anns
                with open(step_path,'w') as fw:
                    json.dump(step_file, fw)
            splits[f'ade20k_incrementalins_{task}_{step}'] = (imgs_dir[9:], step_path[9:])
    splits["ade20k_incrementalins_val"] = (
        "ADEChallengeData2016/images/validation",
        "ADEChallengeData2016/ade20k_instance_val.json")
    return splits

_PREDEFINED_SPLITS = make_incremental_anns()

def _get_ade_instances_meta():
    thing_ids = [k["id"] for k in ADE_CATEGORIES]
    assert len(thing_ids) == 100, len(thing_ids)
    # Mapping from the incontiguous ADE category id to an id in [0, 99]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in ADE_CATEGORIES]
    ret = {
        # "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        # "thing_classes": thing_classes,
    }
    return ret


def load_coco_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str or None): the name of the dataset (e.g., coco_2017_train).
            When provided, this function will also do the following:

            * Put "thing_classes" into the metadata associated with this dataset.
            * Map the category ids into a contiguous range (needed by standard dataset format),
              and add "thing_dataset_id_to_contiguous_id" to the metadata associated
              with this dataset.

            This option should usually be provided, unless users need to load
            the original json content and apply more processing manually.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format (See
        `Using Custom Datasets </tutorials/datasets.html>`_ ) when `dataset_name` is not None.
        If `dataset_name` is None, the returned `category_ids` may be
        incontiguous and may not conform to the Detectron2 standard format.

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if 'val' in dataset_name:
        # ids -> [0,1,...]
        values = list(range(0,len(ids)))
        id_map = dict(zip(ids,values))
        thing_classes = [[cat for cat in ADE_CATEGORIES if cat['id']==i][0]['name'] for i in ids]
    else:
        # ids -> [0,1,...], 0 as backgroud
        dataset, seg, task, step = dataset_name.split('_')
        training_ids = tasks_ade_ins[task][int(step)]
        values = list(range(0,len(training_ids)))
        id_map = dict(zip(training_ids,values))
        thing_classes = [[cat for cat in ADE_CATEGORIES if cat['id']==i][0]['name'] for i in training_ids]

    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        meta.thing_classes = thing_classes
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}
            if "bbox" in obj and len(obj["bbox"]) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                )

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                annotation_category_id = obj["category_id"]
                try:
                    obj["category_id"] = id_map[annotation_category_id]
                except KeyError as e:
                    raise KeyError(
                        f"Encountered category_id={annotation_category_id} "
                        "but this id does not exist in 'categories' of the json file."
                    ) from e
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process.  Please "
            "check https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html carefully"
        )
    return dataset_dicts


def register_coco_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


def register_all_ade20k_instance(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_ade_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_ade20k_instance(_root)
