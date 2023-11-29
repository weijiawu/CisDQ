# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances
from mask2former.data.datasets.register_ade20k_incremental import ADE20K_SEM_SEG
from mask2former.data.datasets.register_voc_incremental import VOC_SEM_SEG
import matplotlib.pyplot as plt

__all__ = ["MaskFormerIncrementalDatasetMapper"]

# class MaskFormerIncrementalDatasetMapper:
#     """
#     A callable which takes a dataset dict in Detectron2 Dataset format,
#     and map it into a format used by MaskFormer for semantic segmentation.
#
#     The callable currently does the following:
#
#     1. Read the image from "file_name"
#     2. Applies geometric transforms to the image and annotation
#     3. Find and applies suitable cropping to the image and annotation
#     4. Prepare image and annotation to Tensors
#     """
#
#     @configurable
#     def __init__(
#         self,
#         classes=None,
#         is_train=True,
#         *,
#         augmentations,
#         image_format,
#         ignore_label,
#         size_divisibility,
#         training_classes,
#         task,
#         step,
#     ):
#         """
#         NOTE: this interface is experimental.
#         Args:
#             is_train: for training or inference
#             augmentations: a list of augmentations or deterministic transforms to apply
#             image_format: an image format supported by :func:`detection_utils.read_image`.
#             ignore_label: the label that is ignored to evaluation
#             size_divisibility: pad image size to be divisible by this value
#         """
#         self.is_train = is_train
#         self.tfm_gens = augmentations
#         self.img_format = image_format
#         self.ignore_label = ignore_label
#         self.size_divisibility = size_divisibility
#         self.training_classes = training_classes
#         self.task = task
#         self.step = step
#         self.background_label = 0
#
#         # map the class label to [1,2,3,...,num_classes]
#         num_classes = len(self.training_classes)
#         # leave 0 as the background class
#         self.class_mapper = list(zip(self.training_classes,range(1,num_classes+1)))
#         self.class_mapper = dict(self.class_mapper)
#         self.reverse_class_mapper = [(v,k) for k,v in self.class_mapper.items()]
#         self.reverse_class_mapper = dict(self.reverse_class_mapper)
#         self.classes = classes
#
#
#         incremental_info = '#'*50+'\n'+f'Training step {self.step} of task {self.task}'+'\n'+'#'*50
#         print(incremental_info)
#
#         logger = logging.getLogger(__name__)
#         mode = "training" if is_train else "inference"
#         logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")
#
#     @classmethod
#     def from_config(cls, cfg, is_train=True):
#         # Build augmentation
#         augs = [
#             T.ResizeShortestEdge(
#                 cfg.INPUT.MIN_SIZE_TRAIN,
#                 cfg.INPUT.MAX_SIZE_TRAIN,
#                 cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
#             )
#         ]
#         if cfg.INPUT.CROP.ENABLED:
#             augs.append(
#                 T.RandomCrop_CategoryAreaConstraint(
#                     cfg.INPUT.CROP.TYPE,
#                     cfg.INPUT.CROP.SIZE,
#                     cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
#                     cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
#                 )
#             )
#         if cfg.INPUT.COLOR_AUG_SSD:
#             augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
#         augs.append(T.RandomFlip())
#
#         # Assume always applies to the training set.
#         dataset_names = cfg.DATASETS.TRAIN
#         meta = MetadataCatalog.get(dataset_names[0])
#         ignore_label = meta.ignore_label
#         training_classes = meta.training_classes
#         task = meta.task
#         step = meta.step
#
#         if 'ade20k' in dataset_names[0]:
#             classes = ADE20K_SEM_SEG
#         elif 'voc' in dataset_names[0]:
#             classes = VOC_SEM_SEG
#         else:
#             raise ValueError('no dataset')
#
#         ret = {
#             "classes": classes,
#             "is_train": is_train,
#             "augmentations": augs,
#             "image_format": cfg.INPUT.FORMAT,
#             "ignore_label": ignore_label,
#             "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
#             "training_classes": training_classes,
#             "task": task,
#             "step": step
#         }
#         return ret
#
#     def __call__(self, dataset_dict):
#         """
#         Args:
#             dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
#
#         Returns:
#             dict: a format that builtin models in detectron2 accept
#         """
#         assert self.is_train, "MaskFormerSemanticDatasetMapper should only be used for training!"
#
#         dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
#         image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
#         utils.check_image_size(dataset_dict, image)
#
#         if "sem_seg_file_name" in dataset_dict:
#             # PyTorch transformation not implemented for uint16, so converting it to double first
#             sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
#         else:
#             sem_seg_gt = None
#
#         if sem_seg_gt is None:
#             raise ValueError(
#                 "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
#                     dataset_dict["file_name"]
#                 )
#             )
#
#         # if there is no valid classes after transform, retry transform
#         check = False
#         times = 30
#         t = 0
#         while not check and t<times:
#             aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
#             aug_input_trans, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
#             tmp = np.array(aug_input_trans.sem_seg)
#             classes = np.unique(tmp).astype(np.long)
#             for cls in classes:
#                 if cls in self.training_classes:
#                     check = True
#                     break
#             t =t +1
#             # if not check:
#             #     self.tfm_gens = self.tfm_gens.remove(self.tfm_gens[1])
#         image = aug_input_trans.image
#         sem_seg_gt = aug_input_trans.sem_seg
#
#
#
#         # Pad image and segmentation label here!
#         image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
#         if sem_seg_gt is not None:
#             sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
#
#         if self.size_divisibility > 0:
#             image_size = (image.shape[-2], image.shape[-1])
#             padding_size = [
#                 0,
#                 self.size_divisibility - image_size[1],
#                 0,
#                 self.size_divisibility - image_size[0],
#             ]
#             image = F.pad(image, padding_size, value=128).contiguous()
#             if sem_seg_gt is not None:
#                 sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()
#
#         # ignore (255) 不参与训练的类别
#         tmp = np.array(sem_seg_gt)
#         classes = np.unique(tmp)
#         for cls in classes:
#             if cls == self.ignore_label:
#                 continue
#             if cls not in self.training_classes:
#                 mask = cls == sem_seg_gt
#                 sem_seg_gt[mask] = self.ignore_label
#
#
#         image_shape = (image.shape[-2], image.shape[-1])  # h, w
#
#         # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
#         # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
#         # Therefore it's important to use torch.Tensor.
#         dataset_dict["image"] = image
#
#         if sem_seg_gt is not None:
#             dataset_dict["sem_seg"] = sem_seg_gt.long()
#
#         if "annotations" in dataset_dict:
#             raise ValueError("Semantic segmentation dataset should not have 'annotations'.")
#
#         # Prepare per-category binary masks
#         if sem_seg_gt is not None:
#             sem_seg_gt = sem_seg_gt.numpy()
#             instances = Instances(image_shape)
#             classes = np.unique(sem_seg_gt)
#             # remove ignored region
#             classes = classes[classes != self.ignore_label]
#
#             # map the class inds to [0 1 2 ... num_classes+1]
#             # where 0 is the background class
#             mapper_classes = [self.class_mapper.get(el,0) for el in classes.tolist()]
#
#             instances.gt_classes = torch.tensor(mapper_classes, dtype=torch.int64)
#
#             masks = []
#             for class_id in classes:
#                 masks.append(sem_seg_gt == class_id)
#
#             if len(masks) == 0:
#                 # Some image does not have annotation (all ignored)
#                 instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
#             else:
#                 masks = BitMasks(
#                     torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
#                 )
#                 instances.gt_masks = masks.tensor
#
#             dataset_dict["instances"] = instances
#             dataset_dict["classes_str"] = [self.classes[el] for el in classes]
#
#
#         dataset_dict["task"] = self.task
#         dataset_dict["step"] = self.step
#
#         return dataset_dict
#
#     def viz(self,dataset_dict,idx=0):
#         import matplotlib.pyplot as plt
#         image = dataset_dict['image'][idx].permute(1,2,0).cpu()
#         captions = dataset_dict['captions']
#         for caption,mask in zip(captions,dataset_dict['instances'][idx].gt_masks.tensor):
#             tmp = copy.deepcopy(image)
#             tmp = tmp + mask[:,:,None]*torch.tensor([0,100,0]).to(torch.int8)
#             plt.title(caption)
#             plt.imshow(tmp.cpu())
#             plt.show()
#         print(1)

# class MaskFormerIncrementalDatasetMapper:
#     """
#     A callable which takes a dataset dict in Detectron2 Dataset format,
#     and map it into a format used by MaskFormer for semantic segmentation.
#
#     The callable currently does the following:
#
#     1. Read the image from "file_name"
#     2. Applies geometric transforms to the image and annotation
#     3. Find and applies suitable cropping to the image and annotation
#     4. Prepare image and annotation to Tensors
#     """
#
#     @configurable
#     def __init__(
#         self,
#         classes=None,
#         is_train=True,
#         *,
#         augmentations,
#         image_format,
#         ignore_label,
#         size_divisibility,
#         training_classes,
#         task,
#         step,
#     ):
#         """
#         NOTE: this interface is experimental.
#         Args:
#             is_train: for training or inference
#             augmentations: a list of augmentations or deterministic transforms to apply
#             image_format: an image format supported by :func:`detection_utils.read_image`.
#             ignore_label: the label that is ignored to evaluation
#             size_divisibility: pad image size to be divisible by this value
#         """
#         self.is_train = is_train
#         self.tfm_gens = augmentations
#         self.img_format = image_format
#         self.ignore_label = ignore_label
#         self.size_divisibility = size_divisibility
#         self.training_classes = training_classes
#         self.task = task
#         self.step = step
#         self.background_label = 255
#
#         # map the class label to [1,2,3,...,num_classes]
#         num_classes = len(self.training_classes)
#         # leave 0 as the background class
#         self.class_mapper = list(zip(self.training_classes,range(1,num_classes+1)))
#         self.class_mapper = dict(self.class_mapper)
#         self.reverse_class_mapper = [(v,k) for k,v in self.class_mapper.items()]
#         self.reverse_class_mapper = dict(self.reverse_class_mapper)
#         self.classes = classes
#
#
#         incremental_info = '#'*50+'\n'+f'Training step {self.step} of task {self.task}'+'\n'+'#'*50
#         print(incremental_info)
#
#         logger = logging.getLogger(__name__)
#         mode = "training" if is_train else "inference"
#         logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")
#
#     @classmethod
#     def from_config(cls, cfg, is_train=True):
#         # Build augmentation
#         augs = [
#             T.ResizeShortestEdge(
#                 cfg.INPUT.MIN_SIZE_TRAIN,
#                 cfg.INPUT.MAX_SIZE_TRAIN,
#                 cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
#             )
#         ]
#         if cfg.INPUT.CROP.ENABLED:
#             augs.append(
#                 T.RandomCrop_CategoryAreaConstraint(
#                     cfg.INPUT.CROP.TYPE,
#                     cfg.INPUT.CROP.SIZE,
#                     cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
#                     cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
#                 )
#             )
#         if cfg.INPUT.COLOR_AUG_SSD:
#             augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
#         augs.append(T.RandomFlip())
#
#         # Assume always applies to the training set.
#         dataset_names = cfg.DATASETS.TRAIN
#         meta = MetadataCatalog.get(dataset_names[0])
#         ignore_label = meta.ignore_label
#         training_classes = meta.training_classes
#         task = meta.task
#         step = meta.step
#
#         if 'ade20k' in dataset_names[0]:
#             classes = ADE20K_SEM_SEG
#         elif 'voc' in dataset_names[0]:
#             classes = VOC_SEM_SEG
#         else:
#             raise ValueError('no dataset')
#
#         ret = {
#             "classes": classes,
#             "is_train": is_train,
#             "augmentations": augs,
#             "image_format": cfg.INPUT.FORMAT,
#             "ignore_label": ignore_label,
#             "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
#             "training_classes": training_classes,
#             "task": task,
#             "step": step
#         }
#         return ret
#
#     def __call__(self, dataset_dict):
#         """
#         Args:
#             dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
#
#         Returns:
#             dict: a format that builtin models in detectron2 accept
#         """
#         assert self.is_train, "MaskFormerSemanticDatasetMapper should only be used for training!"
#
#         dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
#         image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
#         utils.check_image_size(dataset_dict, image)
#
#         if "sem_seg_file_name" in dataset_dict:
#             # PyTorch transformation not implemented for uint16, so converting it to double first
#             sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
#         else:
#             sem_seg_gt = None
#
#         if sem_seg_gt is None:
#             raise ValueError(
#                 "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
#                     dataset_dict["file_name"]
#                 )
#             )
#
#         # if there is no valid classes after transform, retry transform
#         check = False
#         times = 30
#         t = 0
#         while not check and t<times:
#             aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
#             aug_input_trans, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
#             tmp = np.array(aug_input_trans.sem_seg)
#             classes = np.unique(tmp).astype(np.long)
#             for cls in classes:
#                 if cls in self.training_classes:
#                     check = True
#                     break
#             t =t +1
#             # if not check:
#             #     self.tfm_gens = self.tfm_gens.remove(self.tfm_gens[1])
#         image = aug_input_trans.image
#         sem_seg_gt = aug_input_trans.sem_seg
#
#
#
#         # Pad image and segmentation label here!
#         image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
#         if sem_seg_gt is not None:
#             sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
#
#         if self.size_divisibility > 0:
#             image_size = (image.shape[-2], image.shape[-1])
#             padding_size = [
#                 0,
#                 self.size_divisibility - image_size[1],
#                 0,
#                 self.size_divisibility - image_size[0],
#             ]
#             image = F.pad(image, padding_size, value=128).contiguous()
#             if sem_seg_gt is not None:
#                 sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()
#
#         # ignore (255) 不参与训练的类别
#         tmp = np.array(sem_seg_gt)
#         classes = np.unique(tmp)
#         for cls in classes:
#             if cls == self.ignore_label: # 255
#                 continue
#             if cls not in self.training_classes:
#                 mask = cls == sem_seg_gt
#                 sem_seg_gt[mask] = self.background_label # 200
#
#
#         image_shape = (image.shape[-2], image.shape[-1])  # h, w
#
#         # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
#         # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
#         # Therefore it's important to use torch.Tensor.
#         dataset_dict["image"] = image
#
#         if sem_seg_gt is not None:
#             dataset_dict["sem_seg"] = sem_seg_gt.long()
#
#         if "annotations" in dataset_dict:
#             raise ValueError("Semantic segmentation dataset should not have 'annotations'.")
#
#         # Prepare per-category binary masks
#         if sem_seg_gt is not None:
#             sem_seg_gt = sem_seg_gt.numpy()
#             instances = Instances(image_shape)
#             classes = np.unique(sem_seg_gt)
#             # remove ignored region
#             classes = classes[(classes != self.ignore_label) & (classes != self.background_label)]
#
#             # map the class inds to [0 1 2 ... num_classes+1]
#             # where 0 is the background class
#             mapper_classes = [self.class_mapper.get(el,0) for el in classes.tolist()]
#
#             instances.gt_classes = torch.tensor(mapper_classes, dtype=torch.int64)
#
#             masks = []
#             for class_id in classes:
#                 masks.append(sem_seg_gt == class_id)
#
#             if len(masks) == 0:
#                 # Some image does not have annotation (all ignored)
#                 instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
#             else:
#                 masks = BitMasks(
#                     torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
#                 )
#                 instances.gt_masks = masks.tensor
#
#             dataset_dict["instances"] = instances
#             dataset_dict["classes_str"] = [self.classes[el] for el in classes]
#
#         dataset_dict["ignore"] = dataset_dict['sem_seg']==self.ignore_label
#         dataset_dict["task"] = self.task
#         dataset_dict["step"] = self.step
#
#         return dataset_dict
#
#     def viz(self,dataset_dict,idx=0):
#         import matplotlib.pyplot as plt
#         image = dataset_dict['image'][idx].permute(1,2,0).cpu()
#         captions = dataset_dict['captions']
#         for caption,mask in zip(captions,dataset_dict['instances'][idx].gt_masks.tensor):
#             tmp = copy.deepcopy(image)
#             tmp = tmp + mask[:,:,None]*torch.tensor([0,100,0]).to(torch.int8)
#             plt.title(caption)
#             plt.imshow(tmp.cpu())
#             plt.show()
#         print(1)


# for voc only
class MaskFormerIncrementalDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        classes=None,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
        training_classes,
        task,
        step,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        self.training_classes = training_classes
        self.task = task
        self.step = step
        self.background_label = 255 # refer to the real background

        # map the class label to [1,2,3,...,num_classes]
        num_classes = len(self.training_classes)
        # leave 0 as the background class
        self.class_mapper = list(zip(self.training_classes,range(1,num_classes+1)))
        self.class_mapper = dict(self.class_mapper)
        self.reverse_class_mapper = [(v,k) for k,v in self.class_mapper.items()]
        self.reverse_class_mapper = dict(self.reverse_class_mapper)
        self.classes = classes


        incremental_info = '#'*50+'\n'+f'Training step {self.step} of task {self.task}'+'\n'+'#'*50
        print(incremental_info)

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")



    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label
        training_classes = meta.training_classes
        task = meta.task
        step = meta.step

        if 'ade20k' in dataset_names[0]:
            classes = ADE20K_SEM_SEG
        elif 'voc' in dataset_names[0]:
            classes = VOC_SEM_SEG
        else:
            raise ValueError('no dataset')

        ret = {
            "classes": classes,
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "training_classes": training_classes,
            "task": task,
            "step": step
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "MaskFormerSemanticDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
        else:
            sem_seg_gt = None

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        # if there is no valid classes after transform, retry transform
        check = False
        times = 30
        t = 0
        while not check and t<times:
            aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
            aug_input_trans, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
            tmp = np.array(aug_input_trans.sem_seg)
            classes = np.unique(tmp).astype(np.long)
            for cls in classes:
                if cls in self.training_classes and cls!=0:
                    check = True
                    break
            t =t +1
            # if not check:
            #     self.tfm_gens = self.tfm_gens.remove(self.tfm_gens[1])
        image = aug_input_trans.image
        sem_seg_gt = aug_input_trans.sem_seg



        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

        # ignore (255) 不参与训练的类别
        tmp = np.array(sem_seg_gt)
        classes = np.unique(tmp)
        for cls in classes:
            if cls == self.ignore_label: # 255
                continue
            if cls not in self.training_classes:
                mask = cls == sem_seg_gt
                sem_seg_gt[mask] = 0 # set as background
                # sem_seg_gt[mask] = self.ignore_label # 255


        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        dataset_dict["fg_seg"] = ((sem_seg_gt!=0)&(sem_seg_gt!=self.ignore_label)).long()

        if "annotations" in dataset_dict:
            raise ValueError("Semantic segmentation dataset should not have 'annotations'.")

        # Prepare per-category binary masks
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # remove ignored region
            classes = classes[(classes != self.ignore_label)]
            if self.step!=0:
                classes = classes[(classes != 0)]

            # map the class inds to [0 1 2 ... num_classes+1]
            # where 0 is the background class
            mapper_classes = [self.class_mapper.get(el,0) for el in classes.tolist()]

            instances.gt_classes = torch.tensor(mapper_classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances
            dataset_dict["classes_str"] = [self.classes[el] for el in classes]

        dataset_dict["ignore"] = dataset_dict['sem_seg']==self.ignore_label
        dataset_dict["task"] = self.task
        dataset_dict["step"] = self.step

        return dataset_dict

    def viz(self,dataset_dict,idx=0):
        import matplotlib.pyplot as plt
        image = dataset_dict['image'].permute(1,2,0).cpu()
        captions = dataset_dict['captions']
        for caption,mask in zip(captions,dataset_dict['instances'].gt_masks.tensor):
            tmp = copy.deepcopy(image)
            tmp = tmp + mask[:,:,None]*torch.tensor([0,100,0]).to(torch.int8)
            plt.title(caption)
            plt.imshow(tmp.cpu())
            plt.show()
        print(1)
