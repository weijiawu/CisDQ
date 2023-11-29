# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import cv2
import os

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from params import params
import copy
from mask2former.utils.kd import *

version = params['version']

if version == 'freeze_old':
    @META_ARCH_REGISTRY.register()
    class IncrementalMaskFormer(nn.Module):
        """
        Main class for mask classification semantic segmentation architectures.
        """

        @configurable
        def __init__(
            self,
            *,
            name: str,
            backbone: Backbone,
            sem_seg_head: nn.Module,
            criterion: nn.Module,
            num_queries: int,
            object_mask_threshold: float,
            overlap_threshold: float,
            metadata,
            size_divisibility: int,
            sem_seg_postprocess_before_inference: bool,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            # inference
            semantic_on: bool,
            panoptic_on: bool,
            instance_on: bool,
            test_topk_per_image: int,
        ):
            """
            Args:
                backbone: a backbone module, must follow detectron2's backbone interface
                sem_seg_head: a module that predicts semantic segmentation from backbone features
                criterion: a module that defines the loss
                num_queries: int, number of queries
                object_mask_threshold: float, threshold to filter query based on classification score
                    for panoptic segmentation inference
                overlap_threshold: overlap threshold used in general inference for panoptic segmentation
                metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                    segmentation inference
                size_divisibility: Some backbones require the input height and width to be divisible by a
                    specific integer. We can use this to override such requirement.
                sem_seg_postprocess_before_inference: whether to resize the prediction back
                    to original input size before semantic segmentation inference or after.
                    For high-resolution dataset like Mapillary, resizing predictions before
                    inference will cause OOM error.
                pixel_mean, pixel_std: list or tuple with #channels element, representing
                    the per-channel mean and std to be used to normalize the input image
                semantic_on: bool, whether to output semantic segmentation prediction
                instance_on: bool, whether to output instance segmentation prediction
                panoptic_on: bool, whether to output panoptic segmentation prediction
                test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
            """
            super().__init__()

            dataset, _, task, step = name.split('_')
            self.dataset = dataset
            self.task = task
            self.step = int(step)

            sem_seg_head.predictor.task = self.task
            sem_seg_head.predictor.step = self.step

            self.backbone = backbone
            self.sem_seg_head = sem_seg_head
            self.criterion = criterion
            self.num_queries = num_queries
            self.overlap_threshold = overlap_threshold
            self.object_mask_threshold = object_mask_threshold
            self.metadata = metadata
            if size_divisibility < 0:
                # use backbone size_divisibility if not set
                size_divisibility = self.backbone.size_divisibility
            self.size_divisibility = size_divisibility
            self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
            self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

            # additional args
            self.semantic_on = semantic_on
            self.instance_on = instance_on
            self.panoptic_on = panoptic_on
            self.test_topk_per_image = test_topk_per_image

            if not self.semantic_on:
                assert self.sem_seg_postprocess_before_inference

            # incremental setting
            trainable_params = []
            if self.step!=0:
                for name,param in self.named_parameters():
                    if 'incre' not in name:
                        param.requires_grad = False
                    else:
                        trainable_params.append(name)
                print('#'*50)
                print('Trainable params :')
                print(trainable_params)
                print('#' * 50)


        @classmethod
        def from_config(cls, cfg):
            backbone = build_backbone(cfg)
            sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

            # Loss parameters:
            deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
            no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

            # loss weights
            class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
            dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
            mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

            # building criterion
            matcher = HungarianMatcher(
                cost_class=class_weight,
                cost_mask=mask_weight,
                cost_dice=dice_weight,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            )

            weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

            if deep_supervision:
                dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
                aux_weight_dict = {}
                for i in range(dec_layers - 1):
                    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
                weight_dict.update(aux_weight_dict)

            losses = ["labels", "masks"]

            dataset, seg, task, step = cfg.DATASETS.TRAIN[0].split('_')

            if dataset=='ade20k':
                if seg == 'incrementalins':
                    from mask2former.data.datasets.register_ade20k_incremental_instance import tasks_ade_ins
                    training_classes = tasks_ade_ins[task][int(step)]
                else:
                    from mask2former.data.datasets.register_ade20k_incremental import tasks_ade
                    training_classes = tasks_ade[task][int(step)]
                cr_num_classes = len(training_classes)
            elif dataset=='voc':
                from mask2former.data.datasets.register_voc_incremental import tasks_voc
                training_classes = tasks_voc[task][int(step)]
                cr_num_classes = len(training_classes)
            elif dataset=='coco':
                from mask2former.data.datasets.register_coco_incremental_instance import tasks_coco_ins
                training_classes = tasks_coco_ins[task][int(step)]
                cr_num_classes = len(training_classes)
            else:
                print(f'no dataset {dataset}')
                raise ValueError('No dataset')

            criterion = SetCriterion(
                cr_num_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight,
                losses=losses,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
                oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
                importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            )


            return {
                "name": cfg.DATASETS.TRAIN[0],
                "backbone": backbone,
                "sem_seg_head": sem_seg_head,
                "criterion": criterion,
                "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
                "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
                "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
                "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
                "sem_seg_postprocess_before_inference": (
                    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                    or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                    or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
                ),
                "pixel_mean": cfg.MODEL.PIXEL_MEAN,
                "pixel_std": cfg.MODEL.PIXEL_STD,
                # inference
                "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
                "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
                "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
                "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            }

        @property
        def device(self):
            return self.pixel_mean.device

        def forward(self, batched_inputs):
            """
            Args:
                batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                    Each item in the list contains the inputs for one image.
                    For now, each item in the list is a dict that contains:
                       * "image": Tensor, image in (C, H, W) format.
                       * "instances": per-region ground truth
                       * Other information that's included in the original dicts, such as:
                         "height", "width" (int): the output resolution of the model (may be different
                         from input resolution), used in inference.
            Returns:
                list[dict]:
                    each dict has the results for one image. The dict contains the following keys:

                    * "sem_seg":
                        A Tensor that represents the
                        per-pixel segmentation prediced by the head.
                        The prediction has shape KxHxW that represents the logits of
                        each class for each pixel.
                    * "panoptic_seg":
                        A tuple that represent panoptic output
                        panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                        segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                            Each dict contains keys "id", "category_id", "isthing".
            """
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            features = self.backbone(images.tensor)

            outputs = self.sem_seg_head(features)

            if self.training:
                # mask classification target
                if "instances" in batched_inputs[0]:
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                    gt_ignores = [x["ignore"].to(self.device) for x in batched_inputs]
                    targets = self.prepare_targets(gt_instances, gt_ignores, images)
                else:
                    targets = None

                outputs.pop('kd_features')

                # bipartite matching-based loss
                losses = self.criterion(outputs, targets)

                for k in list(losses.keys()):
                    if k in self.criterion.weight_dict:
                        losses[k] *= self.criterion.weight_dict[k]
                    else:
                        # remove this loss if not specified in `weight_dict`
                        losses.pop(k)
                return losses
            else:
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )

                del outputs

                processed_results = []
                for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                    mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    processed_results.append({})

                    if self.sem_seg_postprocess_before_inference:
                        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            mask_pred_result, image_size, height, width
                        )
                        mask_cls_result = mask_cls_result.to(mask_pred_result)

                    # semantic segmentation inference
                    if self.semantic_on:
                        r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                        if not self.sem_seg_postprocess_before_inference:
                            r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                        processed_results[-1]["sem_seg"] = r

                    # panoptic segmentation inference
                    if self.panoptic_on:
                        panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["panoptic_seg"] = panoptic_r

                    # instance segmentation inference
                    if self.instance_on:
                        instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["instances"] = instance_r

                return processed_results

        def prepare_targets(self, targets, gt_ignores, images):
            h_pad, w_pad = images.tensor.shape[-2:]
            new_targets = []
            for targets_per_image, gt_ignore in zip(targets, gt_ignores):
                # pad gt
                gt_masks = targets_per_image.gt_masks
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
                new_targets.append(
                    {
                        "labels": targets_per_image.gt_classes,
                        "masks": padded_masks,
                        "gt_ignore": gt_ignore,
                    }
                )
            return new_targets

        def semantic_inference(self, mask_cls, mask_pred):
            mask_cls = F.softmax(mask_cls, dim=-1)[..., 1:] # the first class is the background
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg

        def panoptic_inference(self, mask_cls, mask_pred):
            scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
            mask_pred = mask_pred.sigmoid()

            keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[keep]
            cur_mask_cls = mask_cls[keep]
            cur_mask_cls = cur_mask_cls[:, :-1]

            cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

            h, w = cur_masks.shape[-2:]
            panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
            segments_info = []

            current_segment_id = 0

            if cur_masks.shape[0] == 0:
                # We didn't detect any mask :(
                return panoptic_seg, segments_info
            else:
                # take argmax
                cur_mask_ids = cur_prob_masks.argmax(0)
                stuff_memory_list = {}
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()
                    isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                    mask_area = (cur_mask_ids == k).sum().item()
                    original_area = (cur_masks[k] >= 0.5).sum().item()
                    mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        if mask_area / original_area < self.overlap_threshold:
                            continue

                        # merge stuff regions
                        if not isthing:
                            if int(pred_class) in stuff_memory_list.keys():
                                panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                                continue
                            else:
                                stuff_memory_list[int(pred_class)] = current_segment_id + 1

                        current_segment_id += 1
                        panoptic_seg[mask] = current_segment_id

                        segments_info.append(
                            {
                                "id": current_segment_id,
                                "isthing": bool(isthing),
                                "category_id": int(pred_class),
                            }
                        )

                return panoptic_seg, segments_info

        def instance_inference(self, mask_cls, mask_pred):
            # mask_pred is already processed to have the same shape as original input
            image_size = mask_pred.shape[-2:]

            # [Q, K]
            # scores = F.softmax(mask_cls, dim=-1)[:, :-1]
            scores = F.softmax(mask_cls, dim=-1)[:, 1:]
            num_queries, num_classes = scores.shape
            labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
            # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
            labels_per_image = labels[topk_indices]

            topk_indices = topk_indices // num_classes
            # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
            mask_pred = mask_pred[topk_indices]

            # if this is panoptic segmentation, we only keep the "thing" classes
            if self.panoptic_on:
                keep = torch.zeros_like(scores_per_image).bool()
                for i, lab in enumerate(labels_per_image):
                    keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]
                mask_pred = mask_pred[keep]

            result = Instances(image_size)
            # mask (before sigmoid)
            result.pred_masks = (mask_pred > 0).float()
            result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
            # Uncomment the following to get boxes from masks (this is slow)
            # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

            # calculate average mask prob
            mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
            result.scores = scores_per_image * mask_scores_per_image
            result.pred_classes = labels_per_image
            return result

elif version == 'baseline':
    class SoftDiceLoss(nn.Module):
        def __init__(self, weight=None, size_average=True):
            super(SoftDiceLoss, self).__init__()

        def forward(self, logits, targets):
            bs = targets.size(0)
            smooth = 1.

            probs = F.sigmoid(logits)
            m1 = probs.view(bs, -1)
            m2 = targets.view(bs, -1)
            intersection = (m1 * m2)

            score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
            score = 1 - score.sum() / bs
            return score+50.
    @META_ARCH_REGISTRY.register()
    class IncrementalMaskFormer(nn.Module):
        """
        Main class for mask classification semantic segmentation architectures.
        """

        @configurable
        def __init__(
                self,
                *,
                cfg: None,
                name: str,
                backbone: Backbone,
                sem_seg_head: nn.Module,
                criterion: nn.Module,
                num_queries: int,
                object_mask_threshold: float,
                overlap_threshold: float,
                metadata,
                size_divisibility: int,
                sem_seg_postprocess_before_inference: bool,
                pixel_mean: Tuple[float],
                pixel_std: Tuple[float],
                # inference
                semantic_on: bool,
                panoptic_on: bool,
                instance_on: bool,
                test_topk_per_image: int,
        ):
            """
            Args:
                backbone: a backbone module, must follow detectron2's backbone interface
                sem_seg_head: a module that predicts semantic segmentation from backbone features
                criterion: a module that defines the loss
                num_queries: int, number of queries
                object_mask_threshold: float, threshold to filter query based on classification score
                    for panoptic segmentation inference
                overlap_threshold: overlap threshold used in general inference for panoptic segmentation
                metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                    segmentation inference
                size_divisibility: Some backbones require the input height and width to be divisible by a
                    specific integer. We can use this to override such requirement.
                sem_seg_postprocess_before_inference: whether to resize the prediction back
                    to original input size before semantic segmentation inference or after.
                    For high-resolution dataset like Mapillary, resizing predictions before
                    inference will cause OOM error.
                pixel_mean, pixel_std: list or tuple with #channels element, representing
                    the per-channel mean and std to be used to normalize the input image
                semantic_on: bool, whether to output semantic segmentation prediction
                instance_on: bool, whether to output instance segmentation prediction
                panoptic_on: bool, whether to output panoptic segmentation prediction
                test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
            """
            super().__init__()
            self.cfg = cfg
            dataset, _, task, step = name.split('_')
            self.dataset = dataset
            self.task = task
            self.step = int(step)

            sem_seg_head.predictor.task = self.task
            sem_seg_head.predictor.step = self.step

            self.backbone = backbone
            self.sem_seg_head = sem_seg_head
            self.criterion = criterion
            self.num_queries = num_queries
            self.overlap_threshold = overlap_threshold
            self.object_mask_threshold = object_mask_threshold
            self.metadata = metadata
            if size_divisibility < 0:
                # use backbone size_divisibility if not set
                size_divisibility = self.backbone.size_divisibility
            self.size_divisibility = size_divisibility
            self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
            self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

            # additional args
            self.semantic_on = semantic_on
            self.instance_on = instance_on
            self.panoptic_on = panoptic_on
            self.test_topk_per_image = test_topk_per_image

            if not self.semantic_on:
                assert self.sem_seg_postprocess_before_inference

            self.dice_loss = SoftDiceLoss()

            # incremental setting
            trainable_params = []
            if self.step != 0:
                for name, param in self.named_parameters():
                    if 'incre' not in name:
                        param.requires_grad = True
                    else:
                        trainable_params.append(name)
                print('#' * 50)
                print('Trainable params :')
                print(trainable_params)
                print('#' * 50)



        @classmethod
        def from_config(cls, cfg):
            backbone = build_backbone(cfg)
            sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

            # Loss parameters:
            deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
            no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

            # loss weights
            class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
            dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
            mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

            # building criterion
            matcher = HungarianMatcher(
                cost_class=class_weight,
                cost_mask=mask_weight,
                cost_dice=dice_weight,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            )

            weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

            if deep_supervision:
                dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
                aux_weight_dict = {}
                for i in range(dec_layers - 1):
                    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
                weight_dict.update(aux_weight_dict)

            losses = ["labels", "masks"]

            dataset, seg, task, step = cfg.DATASETS.TRAIN[0].split('_')

            if dataset == 'ade20k':
                if seg == 'incrementalins':
                    from mask2former.data.datasets.register_ade20k_incremental_instance import tasks_ade_ins
                    training_classes = tasks_ade_ins[task][int(step)]
                else:
                    from mask2former.data.datasets.register_ade20k_incremental import tasks_ade
                    training_classes = tasks_ade[task][int(step)]
                cr_num_classes = len(training_classes)
            elif dataset == 'voc':
                from mask2former.data.datasets.register_voc_incremental import tasks_voc
                training_classes = tasks_voc[task][int(step)]
                cr_num_classes = len(training_classes)
            elif dataset == 'coco':
                from mask2former.data.datasets.register_coco_incremental_instance import tasks_coco_ins
                training_classes = tasks_coco_ins[task][int(step)]
                cr_num_classes = len(training_classes)
            else:
                print(f'no dataset {dataset}')
                raise ValueError('No dataset')

            criterion = SetCriterion(
                cr_num_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight,
                losses=losses,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
                oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
                importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            )

            return {
                "cfg": cfg,
                "name": cfg.DATASETS.TRAIN[0],
                "backbone": backbone,
                "sem_seg_head": sem_seg_head,
                "criterion": criterion,
                "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
                "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
                "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
                "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
                "sem_seg_postprocess_before_inference": (
                        cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                        or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                        or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
                ),
                "pixel_mean": cfg.MODEL.PIXEL_MEAN,
                "pixel_std": cfg.MODEL.PIXEL_STD,
                # inference
                "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
                "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
                "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
                "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            }

        @property
        def device(self):
            return self.pixel_mean.device

        def losskd(self, pkd_features, kd_features):

            losskd = dict()
            predictions_classs = kd_features['predictions_class']
            ppredictions_classs = pkd_features['predictions_class']
            predictions_masks = kd_features['predictions_mask']
            ppredictions_masks = pkd_features['predictions_mask']
            # nl = list(range(len(predictions_classs)))
            nl = [9]
            nk = predictions_classs[0].keys()
            for l in nl:
                for k in nk:
                    b, q, c = ppredictions_classs[l][k].shape
                    pred = ppredictions_classs[l][k].reshape(-1, c).softmax(-1)
                    gt = predictions_classs[l][k].reshape(-1, c).softmax(-1)
                    losskd[f'loss_kd_class{l}_{k}'] = F.kl_div(pred.log(), gt, reduction='sum') / (b * q) * 100

                    b, q, h, w = predictions_masks[l][k].shape
                    pred = ppredictions_masks[l][k].reshape(-1, h, w)
                    gt = (predictions_masks[l][k].reshape(-1, h, w) > 0).to(torch.int64)
                    losskd[f'loss_kd_mask{l}_{k}'] = self.dice_loss(pred, gt) * 100



            return losskd

        def forward(self, batched_inputs, kd=False):
            """
            Args:
                batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                    Each item in the list contains the inputs for one image.
                    For now, each item in the list is a dict that contains:
                       * "image": Tensor, image in (C, H, W) format.
                       * "instances": per-region ground truth
                       * Other information that's included in the original dicts, such as:
                         "height", "width" (int): the output resolution of the model (may be different
                         from input resolution), used in inference.
            Returns:
                list[dict]:
                    each dict has the results for one image. The dict contains the following keys:

                    * "sem_seg":
                        A Tensor that represents the
                        per-pixel segmentation prediced by the head.
                        The prediction has shape KxHxW that represents the logits of
                        each class for each pixel.
                    * "panoptic_seg":
                        A tuple that represent panoptic output
                        panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                        segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                            Each dict contains keys "id", "category_id", "isthing".
            """
            # if False:
            if self.training and not hasattr(self, 'teacher_model') and self.step != 0:
                # build teacher model
                teacher_model = build_model(self.cfg)
                teacher_model.step = self.step - 1
                teacher_model.sem_seg_head.predictor.step = self.step - 1
                DetectionCheckpointer(teacher_model, save_dir='./cache/debug').resume_or_load(
                    self.cfg.MODEL.WEIGHTS
                )
                teacher_model.eval()
                self.teacher_model = teacher_model
            if self.training and hasattr(self, 'teacher_model'):
                kd_features = self.teacher_model(batched_inputs, kd=True)
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            features = self.backbone(images.tensor)

            outputs = self.sem_seg_head(features)

            if kd:
                kd_features = outputs['kd_features']
                kd_features['res_features'] = features
                return kd_features

            if self.training:
                # mask classification target
                if "instances" in batched_inputs[0]:
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                    gt_ignores = [x["ignore"].to(self.device) for x in batched_inputs]
                    targets = self.prepare_targets(gt_instances, gt_ignores, images)
                else:
                    targets = None

                losses = dict()
                if hasattr(self, 'teacher_model'):
                    pkd_features = outputs['kd_features']
                    pkd_features['res_features'] = features
                    losses.update(self.losskd(pkd_features, kd_features))
                if 'kd_features' in outputs:
                    outputs.pop('kd_features')
                # bipartite matching-based loss
                losses.update(self.criterion(outputs, targets))

                for k in list(losses.keys()):
                    if k in self.criterion.weight_dict:
                        losses[k] *= self.criterion.weight_dict[k]
                    else:
                        # remove this loss if not specified in `weight_dict`
                        continue
                        # losses.pop(k)
                return losses
            else:
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )

                del outputs

                processed_results = []
                for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                        mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    processed_results.append({})

                    if self.sem_seg_postprocess_before_inference:
                        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            mask_pred_result, image_size, height, width
                        )
                        mask_cls_result = mask_cls_result.to(mask_pred_result)

                    # semantic segmentation inference
                    if self.semantic_on:
                        r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                        if not self.sem_seg_postprocess_before_inference:
                            r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                        processed_results[-1]["sem_seg"] = r

                    # panoptic segmentation inference
                    if self.panoptic_on:
                        panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["panoptic_seg"] = panoptic_r

                    # instance segmentation inference
                    if self.instance_on:
                        instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["instances"] = instance_r

                return processed_results

        def prepare_targets(self, targets, gt_ignores, images):
            h_pad, w_pad = images.tensor.shape[-2:]
            new_targets = []
            for targets_per_image, gt_ignore in zip(targets, gt_ignores):
                # pad gt
                gt_masks = targets_per_image.gt_masks
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype,
                                           device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
                new_targets.append(
                    {
                        "labels": targets_per_image.gt_classes,
                        "masks": padded_masks,
                        "gt_ignore": gt_ignore,
                    }
                )
            return new_targets

        def semantic_inference(self, mask_cls, mask_pred):
            mask_cls = F.softmax(mask_cls, dim=-1)[..., 1:]  # the first class is the background
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg

        def panoptic_inference(self, mask_cls, mask_pred):
            scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
            mask_pred = mask_pred.sigmoid()

            keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[keep]
            cur_mask_cls = mask_cls[keep]
            cur_mask_cls = cur_mask_cls[:, :-1]

            cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

            h, w = cur_masks.shape[-2:]
            panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
            segments_info = []

            current_segment_id = 0

            if cur_masks.shape[0] == 0:
                # We didn't detect any mask :(
                return panoptic_seg, segments_info
            else:
                # take argmax
                cur_mask_ids = cur_prob_masks.argmax(0)
                stuff_memory_list = {}
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()
                    isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                    mask_area = (cur_mask_ids == k).sum().item()
                    original_area = (cur_masks[k] >= 0.5).sum().item()
                    mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        if mask_area / original_area < self.overlap_threshold:
                            continue

                        # merge stuff regions
                        if not isthing:
                            if int(pred_class) in stuff_memory_list.keys():
                                panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                                continue
                            else:
                                stuff_memory_list[int(pred_class)] = current_segment_id + 1

                        current_segment_id += 1
                        panoptic_seg[mask] = current_segment_id

                        segments_info.append(
                            {
                                "id": current_segment_id,
                                "isthing": bool(isthing),
                                "category_id": int(pred_class),
                            }
                        )

                return panoptic_seg, segments_info

        def instance_inference(self, mask_cls, mask_pred):
            # mask_pred is already processed to have the same shape as original input
            image_size = mask_pred.shape[-2:]

            # [Q, K]
            # scores = F.softmax(mask_cls, dim=-1)[:, :-1]
            scores = F.softmax(mask_cls, dim=-1)[:, 1:]
            num_queries, num_classes = scores.shape
            labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
            # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
            labels_per_image = labels[topk_indices]

            topk_indices = topk_indices // num_classes
            # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
            mask_pred = mask_pred[topk_indices]

            # if this is panoptic segmentation, we only keep the "thing" classes
            if self.panoptic_on:
                keep = torch.zeros_like(scores_per_image).bool()
                for i, lab in enumerate(labels_per_image):
                    keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]
                mask_pred = mask_pred[keep]

            result = Instances(image_size)
            # mask (before sigmoid)
            result.pred_masks = (mask_pred > 0).float()
            result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
            # Uncomment the following to get boxes from masks (this is slow)
            # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

            # calculate average mask prob
            mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
                    result.pred_masks.flatten(1).sum(1) + 1e-6)
            result.scores = scores_per_image * mask_scores_per_image
            result.pred_classes = labels_per_image
            return result

elif version == 'qfl':

    class QFocalLoss(nn.Module):
        def __init__(self, weight=None, size_average=True):
            super(QFocalLoss, self).__init__()

        def bce_loss(self, preds, targets):
            return -(targets*torch.log(preds)+(1-targets)*torch.log(1-preds))

        def forward(self, preds, targets):
            preds = preds.sigmoid()
            preds = torch.clamp(preds, min=1e-3, max=1 - 1e-3)
            targets = targets.sigmoid()
            loss = self.bce_loss(preds, targets)
            loss = (targets-preds)**2*loss
            # loss = -(targets-logits)**2*(targets*torch.log(logits)+(1-targets)*torch.log(1-logits))
            loss = loss.mean()
            return loss
    @META_ARCH_REGISTRY.register()
    class IncrementalMaskFormer(nn.Module):
        """
        Main class for mask classification semantic segmentation architectures.
        """

        @configurable
        def __init__(
                self,
                *,
                cfg: None,
                name: str,
                backbone: Backbone,
                sem_seg_head: nn.Module,
                criterion: nn.Module,
                num_queries: int,
                object_mask_threshold: float,
                overlap_threshold: float,
                metadata,
                size_divisibility: int,
                sem_seg_postprocess_before_inference: bool,
                pixel_mean: Tuple[float],
                pixel_std: Tuple[float],
                # inference
                semantic_on: bool,
                panoptic_on: bool,
                instance_on: bool,
                test_topk_per_image: int,
        ):
            """
            Args:
                backbone: a backbone module, must follow detectron2's backbone interface
                sem_seg_head: a module that predicts semantic segmentation from backbone features
                criterion: a module that defines the loss
                num_queries: int, number of queries
                object_mask_threshold: float, threshold to filter query based on classification score
                    for panoptic segmentation inference
                overlap_threshold: overlap threshold used in general inference for panoptic segmentation
                metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                    segmentation inference
                size_divisibility: Some backbones require the input height and width to be divisible by a
                    specific integer. We can use this to override such requirement.
                sem_seg_postprocess_before_inference: whether to resize the prediction back
                    to original input size before semantic segmentation inference or after.
                    For high-resolution dataset like Mapillary, resizing predictions before
                    inference will cause OOM error.
                pixel_mean, pixel_std: list or tuple with #channels element, representing
                    the per-channel mean and std to be used to normalize the input image
                semantic_on: bool, whether to output semantic segmentation prediction
                instance_on: bool, whether to output instance segmentation prediction
                panoptic_on: bool, whether to output panoptic segmentation prediction
                test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
            """
            super().__init__()
            self.cfg = cfg
            dataset, _, task, step = name.split('_')
            self.dataset = dataset
            self.task = task
            self.step = int(step)

            sem_seg_head.predictor.task = self.task
            sem_seg_head.predictor.step = self.step

            self.backbone = backbone
            self.sem_seg_head = sem_seg_head
            self.criterion = criterion
            self.num_queries = num_queries
            self.overlap_threshold = overlap_threshold
            self.object_mask_threshold = object_mask_threshold
            self.metadata = metadata
            if size_divisibility < 0:
                # use backbone size_divisibility if not set
                size_divisibility = self.backbone.size_divisibility
            self.size_divisibility = size_divisibility
            self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
            self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

            # additional args
            self.semantic_on = semantic_on
            self.instance_on = instance_on
            self.panoptic_on = panoptic_on
            self.test_topk_per_image = test_topk_per_image

            if not self.semantic_on:
                assert self.sem_seg_postprocess_before_inference

            self.qfl_loss = QFocalLoss()

            # incremental setting
            trainable_params = []
            if self.step != 0:
                for name, param in self.named_parameters():
                    if 'incre' not in name:
                        param.requires_grad = True
                    else:
                        trainable_params.append(name)
                print('#' * 50)
                print('Trainable params :')
                print(trainable_params)
                print('#' * 50)

        @classmethod
        def from_config(cls, cfg):
            backbone = build_backbone(cfg)
            sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

            # Loss parameters:
            deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
            no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

            # loss weights
            class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
            dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
            mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

            # building criterion
            matcher = HungarianMatcher(
                cost_class=class_weight,
                cost_mask=mask_weight,
                cost_dice=dice_weight,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            )

            weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

            if deep_supervision:
                dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
                aux_weight_dict = {}
                for i in range(dec_layers - 1):
                    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
                weight_dict.update(aux_weight_dict)

            losses = ["labels", "masks"]

            dataset, seg, task, step = cfg.DATASETS.TRAIN[0].split('_')

            if dataset == 'ade20k':
                if seg == 'incrementalins':
                    from mask2former.data.datasets.register_ade20k_incremental_instance import tasks_ade_ins
                    training_classes = tasks_ade_ins[task][int(step)]
                else:
                    from mask2former.data.datasets.register_ade20k_incremental import tasks_ade
                    training_classes = tasks_ade[task][int(step)]
                cr_num_classes = len(training_classes)
            elif dataset == 'voc':
                from mask2former.data.datasets.register_voc_incremental import tasks_voc
                training_classes = tasks_voc[task][int(step)]
                cr_num_classes = len(training_classes)
            elif dataset == 'coco':
                from mask2former.data.datasets.register_coco_incremental_instance import tasks_coco_ins
                training_classes = tasks_coco_ins[task][int(step)]
                cr_num_classes = len(training_classes)
            else:
                print(f'no dataset {dataset}')
                raise ValueError('No dataset')

            criterion = SetCriterion(
                cr_num_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight,
                losses=losses,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
                oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
                importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            )

            return {
                "cfg": cfg,
                "name": cfg.DATASETS.TRAIN[0],
                "backbone": backbone,
                "sem_seg_head": sem_seg_head,
                "criterion": criterion,
                "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
                "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
                "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
                "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
                "sem_seg_postprocess_before_inference": (
                        cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                        or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                        or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
                ),
                "pixel_mean": cfg.MODEL.PIXEL_MEAN,
                "pixel_std": cfg.MODEL.PIXEL_STD,
                # inference
                "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
                "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
                "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
                "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            }

        @property
        def device(self):
            return self.pixel_mean.device

        def losskd(self, pkd_features, kd_features):
            losskd = dict()
            predictions_classs = kd_features['predictions_class']
            ppredictions_classs = pkd_features['predictions_class']
            predictions_masks = kd_features['predictions_mask']
            ppredictions_masks = pkd_features['predictions_mask']
            # nl = list(range(len(predictions_classs)))
            nl = [9]
            nk = predictions_classs[0].keys()
            for l in nl:
                for k in nk:
                    b, q, c = ppredictions_classs[l][k].shape
                    pred = ppredictions_classs[l][k].reshape(-1, c).softmax(-1)
                    gt = predictions_classs[l][k].reshape(-1, c).softmax(-1)
                    losskd[f'loss_kd_class{l}_{k}'] = F.kl_div(pred.log(), gt, reduction='sum') / (b * q) * 100

                    pos = ppredictions_classs[l][k].reshape(-1, c).argmax(-1) > 0

                    if pos.sum() < 1:
                        losskd[f'loss_kd_mask{l}_{k}'] = pred.sum() * 0
                        continue

                    b, q, h, w = predictions_masks[l][k].shape
                    pred = ppredictions_masks[l][k].reshape(-1,h,w)[pos]
                    gt = predictions_masks[l][k].reshape(-1,h,w)[pos]
                    # with torch.cuda.amp.autocast(enabled=False):
                    losskd[f'loss_kd_mask{l}_{k}'] = self.qfl_loss(pred, gt)*100

            return losskd

        def detach_features(self, features):
            if isinstance(features,torch.Tensor):
                return features.detach()
            elif isinstance(features,dict):
                new_dict = dict()
                for k,v in features.items():
                    new_dict[k] = self.detach_features(v)
                return new_dict
            elif isinstance(features,list):
                return [self.detach_features(el) for el in features]
            else:
                raise ValueError("unknow type")

        def forward(self, batched_inputs, kd=False):
            """
            Args:
                batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                    Each item in the list contains the inputs for one image.
                    For now, each item in the list is a dict that contains:
                       * "image": Tensor, image in (C, H, W) format.
                       * "instances": per-region ground truth
                       * Other information that's included in the original dicts, such as:
                         "height", "width" (int): the output resolution of the model (may be different
                         from input resolution), used in inference.
            Returns:
                list[dict]:
                    each dict has the results for one image. The dict contains the following keys:

                    * "sem_seg":
                        A Tensor that represents the
                        per-pixel segmentation prediced by the head.
                        The prediction has shape KxHxW that represents the logits of
                        each class for each pixel.
                    * "panoptic_seg":
                        A tuple that represent panoptic output
                        panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                        segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                            Each dict contains keys "id", "category_id", "isthing".
            """
            # if False:
            if self.training and not hasattr(self, 'teacher_model') and self.step != 0:
                # build teacher model
                teacher_model = build_model(self.cfg)
                teacher_model.step = self.step - 1
                teacher_model.sem_seg_head.predictor.step = self.step - 1
                DetectionCheckpointer(teacher_model, save_dir='./cache/debug').resume_or_load(
                    self.cfg.MODEL.WEIGHTS
                )
                teacher_model.eval()
                self.teacher_model = teacher_model
            if self.training and hasattr(self, 'teacher_model'):
                kd_features = self.teacher_model(batched_inputs, kd=True)
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            features = self.backbone(images.tensor)

            outputs = self.sem_seg_head(features)

            if kd:
                kd_features = outputs['kd_features']
                kd_features['res_features'] = features

                return self.detach_features(kd_features)

            if self.training:
                # mask classification target
                if "instances" in batched_inputs[0]:
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                    gt_ignores = [x["ignore"].to(self.device) for x in batched_inputs]
                    targets = self.prepare_targets(gt_instances, gt_ignores, images)
                else:
                    targets = None

                losses = dict()
                if hasattr(self, 'teacher_model'):
                    pkd_features = outputs['kd_features']
                    pkd_features['res_features'] = features
                    losses.update(self.losskd(pkd_features, kd_features))
                if 'kd_features' in outputs:
                    outputs.pop('kd_features')
                # bipartite matching-based loss
                losses.update(self.criterion(outputs, targets))

                for k in list(losses.keys()):
                    if k in self.criterion.weight_dict:
                        losses[k] *= self.criterion.weight_dict[k]
                    else:
                        # remove this loss if not specified in `weight_dict`
                        continue
                        # losses.pop(k)
                return losses
            else:
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )

                del outputs

                processed_results = []
                for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                        mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    processed_results.append({})

                    if self.sem_seg_postprocess_before_inference:
                        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            mask_pred_result, image_size, height, width
                        )
                        mask_cls_result = mask_cls_result.to(mask_pred_result)

                    # semantic segmentation inference
                    if self.semantic_on:
                        r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                        if not self.sem_seg_postprocess_before_inference:
                            r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                        processed_results[-1]["sem_seg"] = r

                    # panoptic segmentation inference
                    if self.panoptic_on:
                        panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["panoptic_seg"] = panoptic_r

                    # instance segmentation inference
                    if self.instance_on:
                        instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["instances"] = instance_r

                return processed_results

        def prepare_targets(self, targets, gt_ignores, images):
            h_pad, w_pad = images.tensor.shape[-2:]
            new_targets = []
            for targets_per_image, gt_ignore in zip(targets, gt_ignores):
                # pad gt
                gt_masks = targets_per_image.gt_masks
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype,
                                           device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
                new_targets.append(
                    {
                        "labels": targets_per_image.gt_classes,
                        "masks": padded_masks,
                        "gt_ignore": gt_ignore,
                    }
                )
            return new_targets

        def semantic_inference(self, mask_cls, mask_pred):
            mask_cls = F.softmax(mask_cls, dim=-1)[..., 1:]  # the first class is the background
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg

        def panoptic_inference(self, mask_cls, mask_pred):
            scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
            mask_pred = mask_pred.sigmoid()

            keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[keep]
            cur_mask_cls = mask_cls[keep]
            cur_mask_cls = cur_mask_cls[:, :-1]

            cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

            h, w = cur_masks.shape[-2:]
            panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
            segments_info = []

            current_segment_id = 0

            if cur_masks.shape[0] == 0:
                # We didn't detect any mask :(
                return panoptic_seg, segments_info
            else:
                # take argmax
                cur_mask_ids = cur_prob_masks.argmax(0)
                stuff_memory_list = {}
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()
                    isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                    mask_area = (cur_mask_ids == k).sum().item()
                    original_area = (cur_masks[k] >= 0.5).sum().item()
                    mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        if mask_area / original_area < self.overlap_threshold:
                            continue

                        # merge stuff regions
                        if not isthing:
                            if int(pred_class) in stuff_memory_list.keys():
                                panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                                continue
                            else:
                                stuff_memory_list[int(pred_class)] = current_segment_id + 1

                        current_segment_id += 1
                        panoptic_seg[mask] = current_segment_id

                        segments_info.append(
                            {
                                "id": current_segment_id,
                                "isthing": bool(isthing),
                                "category_id": int(pred_class),
                            }
                        )

                return panoptic_seg, segments_info

        def instance_inference(self, mask_cls, mask_pred):
            # mask_pred is already processed to have the same shape as original input
            image_size = mask_pred.shape[-2:]

            # [Q, K]
            # scores = F.softmax(mask_cls, dim=-1)[:, :-1]
            scores = F.softmax(mask_cls, dim=-1)[:, 1:]
            num_queries, num_classes = scores.shape
            labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
            # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
            labels_per_image = labels[topk_indices]

            topk_indices = topk_indices // num_classes
            # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
            mask_pred = mask_pred[topk_indices]

            # if this is panoptic segmentation, we only keep the "thing" classes
            if self.panoptic_on:
                keep = torch.zeros_like(scores_per_image).bool()
                for i, lab in enumerate(labels_per_image):
                    keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]
                mask_pred = mask_pred[keep]

            result = Instances(image_size)
            # mask (before sigmoid)
            result.pred_masks = (mask_pred > 0).float()
            result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
            # Uncomment the following to get boxes from masks (this is slow)
            # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

            # calculate average mask prob
            mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
                    result.pred_masks.flatten(1).sum(1) + 1e-6)
            result.scores = scores_per_image * mask_scores_per_image
            result.pred_classes = labels_per_image
            return result

elif version == 'qkd':
    class SoftDiceLoss(nn.Module):
        def __init__(self, weight=None, size_average=True):
            super(SoftDiceLoss, self).__init__()

        def forward(self, logits, targets):
            bs = targets.size(0)
            smooth = 1.

            probs = F.sigmoid(logits)
            m1 = probs.view(bs, -1)
            m2 = targets.view(bs, -1)
            intersection = (m1 * m2)

            score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
            score = 1 - score.sum() / bs
            return score
    @META_ARCH_REGISTRY.register()
    class IncrementalMaskFormer(nn.Module):
        """
        Main class for mask classification semantic segmentation architectures.
        """

        @configurable
        def __init__(
                self,
                *,
                cfg: None,
                name: str,
                backbone: Backbone,
                sem_seg_head: nn.Module,
                criterion: nn.Module,
                num_queries: int,
                object_mask_threshold: float,
                overlap_threshold: float,
                metadata,
                size_divisibility: int,
                sem_seg_postprocess_before_inference: bool,
                pixel_mean: Tuple[float],
                pixel_std: Tuple[float],
                # inference
                semantic_on: bool,
                panoptic_on: bool,
                instance_on: bool,
                test_topk_per_image: int,
        ):
            """
            Args:
                backbone: a backbone module, must follow detectron2's backbone interface
                sem_seg_head: a module that predicts semantic segmentation from backbone features
                criterion: a module that defines the loss
                num_queries: int, number of queries
                object_mask_threshold: float, threshold to filter query based on classification score
                    for panoptic segmentation inference
                overlap_threshold: overlap threshold used in general inference for panoptic segmentation
                metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                    segmentation inference
                size_divisibility: Some backbones require the input height and width to be divisible by a
                    specific integer. We can use this to override such requirement.
                sem_seg_postprocess_before_inference: whether to resize the prediction back
                    to original input size before semantic segmentation inference or after.
                    For high-resolution dataset like Mapillary, resizing predictions before
                    inference will cause OOM error.
                pixel_mean, pixel_std: list or tuple with #channels element, representing
                    the per-channel mean and std to be used to normalize the input image
                semantic_on: bool, whether to output semantic segmentation prediction
                instance_on: bool, whether to output instance segmentation prediction
                panoptic_on: bool, whether to output panoptic segmentation prediction
                test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
            """
            super().__init__()
            self.cfg = cfg
            dataset, _, task, step = name.split('_')
            self.dataset = dataset
            self.task = task
            self.step = int(step)

            sem_seg_head.predictor.task = self.task
            sem_seg_head.predictor.step = self.step

            self.backbone = backbone
            self.sem_seg_head = sem_seg_head
            self.criterion = criterion
            self.num_queries = num_queries
            self.overlap_threshold = overlap_threshold
            self.object_mask_threshold = object_mask_threshold
            self.metadata = metadata
            if size_divisibility < 0:
                # use backbone size_divisibility if not set
                size_divisibility = self.backbone.size_divisibility
            self.size_divisibility = size_divisibility
            self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
            self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

            # additional args
            self.semantic_on = semantic_on
            self.instance_on = instance_on
            self.panoptic_on = panoptic_on
            self.test_topk_per_image = test_topk_per_image

            if not self.semantic_on:
                assert self.sem_seg_postprocess_before_inference

            self.dice_loss = SoftDiceLoss()

            # incremental setting
            trainable_params = []
            if self.step != 0:
                for name, param in self.named_parameters():
                    if 'incre' not in name:
                        param.requires_grad = True
                    else:
                        trainable_params.append(name)
                print('#' * 50)
                print('Trainable params :')
                print(trainable_params)
                print('#' * 50)

        @classmethod
        def from_config(cls, cfg):
            backbone = build_backbone(cfg)
            sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

            # Loss parameters:
            deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
            no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

            # loss weights
            class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
            dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
            mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

            # building criterion
            matcher = HungarianMatcher(
                cost_class=class_weight,
                cost_mask=mask_weight,
                cost_dice=dice_weight,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            )

            weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

            if deep_supervision:
                dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
                aux_weight_dict = {}
                for i in range(dec_layers - 1):
                    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
                weight_dict.update(aux_weight_dict)

            losses = ["labels", "masks"]

            dataset, seg, task, step = cfg.DATASETS.TRAIN[0].split('_')

            if dataset == 'ade20k':
                if seg == 'incrementalins':
                    from mask2former.data.datasets.register_ade20k_incremental_instance import tasks_ade_ins
                    training_classes = tasks_ade_ins[task][int(step)]
                else:
                    from mask2former.data.datasets.register_ade20k_incremental import tasks_ade
                    training_classes = tasks_ade[task][int(step)]
                cr_num_classes = len(training_classes)
            elif dataset == 'voc':
                from mask2former.data.datasets.register_voc_incremental import tasks_voc
                training_classes = tasks_voc[task][int(step)]
                cr_num_classes = len(training_classes)
            elif dataset == 'coco':
                from mask2former.data.datasets.register_coco_incremental_instance import tasks_coco_ins
                training_classes = tasks_coco_ins[task][int(step)]
                cr_num_classes = len(training_classes)
            else:
                print(f'no dataset {dataset}')
                raise ValueError('No dataset')

            criterion = SetCriterion(
                cr_num_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight,
                losses=losses,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
                oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
                importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            )

            return {
                "cfg": cfg,
                "name": cfg.DATASETS.TRAIN[0],
                "backbone": backbone,
                "sem_seg_head": sem_seg_head,
                "criterion": criterion,
                "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
                "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
                "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
                "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
                "sem_seg_postprocess_before_inference": (
                        cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                        or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                        or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
                ),
                "pixel_mean": cfg.MODEL.PIXEL_MEAN,
                "pixel_std": cfg.MODEL.PIXEL_STD,
                # inference
                "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
                "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
                "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
                "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            }

        @property
        def device(self):
            return self.pixel_mean.device

        def losskd(self, pkd_features, kd_features):
            losskd = dict()
            predictions_classs = kd_features['predictions_class']
            ppredictions_classs = pkd_features['predictions_class']
            predictions_masks = kd_features['predictions_mask']
            ppredictions_masks = pkd_features['predictions_mask']
            # nl = list(range(len(predictions_classs)))
            nl = [9]
            nk = predictions_classs[0].keys()
            for l in nl:
                for k in nk:
                    b, q, c = ppredictions_classs[l][k].shape
                    pred = ppredictions_classs[l][k].reshape(-1, c).softmax(-1)
                    gt = predictions_classs[l][k].reshape(-1, c).softmax(-1)
                    losskd[f'loss_kd_class{l}_{k}'] = F.kl_div(pred.log(), gt, reduction='sum') / (b * q) * 100

                    pos = ppredictions_classs[l][k].reshape(-1, c).argmax(-1) > 0

                    if pos.sum() < 1:
                        losskd[f'loss_kd_mask{l}_{k}'] = pred.sum() * 0
                        continue

                    b, q, h, w = predictions_masks[l][k].shape
                    pred = ppredictions_masks[l][k].reshape(-1, h, w)[pos]
                    gt = (predictions_masks[l][k].reshape(-1, h, w) > 0).to(torch.int64)[pos]
                    losskd[f'loss_kd_mask{l}_{k}'] = self.dice_loss(pred, gt) * 100

            predictions_query = kd_features['predictions_query']
            ppredictions_query = pkd_features['predictions_query']
            nq_previous = len(predictions_query[0])
            loss_qkd = 0
            for query, pred_query in zip(predictions_query, ppredictions_query):
                loss_qkd = (torch.abs(query-pred_query[:nq_previous])/(torch.abs(query)+1e-6)).mean()
            loss_qkd = loss_qkd/len(predictions_query)
            losskd[f'loss_kd_query'] = loss_qkd

            return losskd

        def forward(self, batched_inputs, kd=False):
            """
            Args:
                batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                    Each item in the list contains the inputs for one image.
                    For now, each item in the list is a dict that contains:
                       * "image": Tensor, image in (C, H, W) format.
                       * "instances": per-region ground truth
                       * Other information that's included in the original dicts, such as:
                         "height", "width" (int): the output resolution of the model (may be different
                         from input resolution), used in inference.
            Returns:
                list[dict]:
                    each dict has the results for one image. The dict contains the following keys:

                    * "sem_seg":
                        A Tensor that represents the
                        per-pixel segmentation prediced by the head.
                        The prediction has shape KxHxW that represents the logits of
                        each class for each pixel.
                    * "panoptic_seg":
                        A tuple that represent panoptic output
                        panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                        segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                            Each dict contains keys "id", "category_id", "isthing".
            """
            # if False:
            if self.training and not hasattr(self, 'teacher_model') and self.step != 0:
                # build teacher model
                teacher_model = build_model(self.cfg)
                teacher_model.step = self.step - 1
                teacher_model.sem_seg_head.predictor.step = self.step - 1
                DetectionCheckpointer(teacher_model, save_dir='./cache/debug').resume_or_load(
                    self.cfg.MODEL.WEIGHTS
                )
                teacher_model.eval()
                self.teacher_model = teacher_model
            if self.training and hasattr(self, 'teacher_model'):
                kd_features = self.teacher_model(batched_inputs, kd=True)
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            features = self.backbone(images.tensor)

            outputs = self.sem_seg_head(features)

            if kd:
                kd_features = outputs['kd_features']
                kd_features['res_features'] = features
                return kd_features

            if self.training:
                # mask classification target
                if "instances" in batched_inputs[0]:
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                    gt_ignores = [x["ignore"].to(self.device) for x in batched_inputs]
                    targets = self.prepare_targets(gt_instances, gt_ignores, images)
                else:
                    targets = None

                losses = dict()
                if hasattr(self, 'teacher_model'):
                    pkd_features = outputs['kd_features']
                    pkd_features['res_features'] = features
                    losses.update(self.losskd(pkd_features, kd_features))
                if 'kd_features' in outputs:
                    outputs.pop('kd_features')
                # bipartite matching-based loss
                losses.update(self.criterion(outputs, targets))

                for k in list(losses.keys()):
                    if k in self.criterion.weight_dict:
                        losses[k] *= self.criterion.weight_dict[k]
                    else:
                        # remove this loss if not specified in `weight_dict`
                        continue
                        # losses.pop(k)
                return losses
            else:
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )

                del outputs

                processed_results = []
                for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                        mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    processed_results.append({})

                    if self.sem_seg_postprocess_before_inference:
                        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            mask_pred_result, image_size, height, width
                        )
                        mask_cls_result = mask_cls_result.to(mask_pred_result)

                    # semantic segmentation inference
                    if self.semantic_on:
                        r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                        if not self.sem_seg_postprocess_before_inference:
                            r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                        processed_results[-1]["sem_seg"] = r

                    # panoptic segmentation inference
                    if self.panoptic_on:
                        panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["panoptic_seg"] = panoptic_r

                    # instance segmentation inference
                    if self.instance_on:
                        instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["instances"] = instance_r

                return processed_results

        def prepare_targets(self, targets, gt_ignores, images):
            h_pad, w_pad = images.tensor.shape[-2:]
            new_targets = []
            for targets_per_image, gt_ignore in zip(targets, gt_ignores):
                # pad gt
                gt_masks = targets_per_image.gt_masks
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype,
                                           device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
                new_targets.append(
                    {
                        "labels": targets_per_image.gt_classes,
                        "masks": padded_masks,
                        "gt_ignore": gt_ignore,
                    }
                )
            return new_targets

        def semantic_inference(self, mask_cls, mask_pred):
            mask_cls = F.softmax(mask_cls, dim=-1)[..., 1:]  # the first class is the background
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg

        def panoptic_inference(self, mask_cls, mask_pred):
            scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
            mask_pred = mask_pred.sigmoid()

            keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[keep]
            cur_mask_cls = mask_cls[keep]
            cur_mask_cls = cur_mask_cls[:, :-1]

            cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

            h, w = cur_masks.shape[-2:]
            panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
            segments_info = []

            current_segment_id = 0

            if cur_masks.shape[0] == 0:
                # We didn't detect any mask :(
                return panoptic_seg, segments_info
            else:
                # take argmax
                cur_mask_ids = cur_prob_masks.argmax(0)
                stuff_memory_list = {}
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()
                    isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                    mask_area = (cur_mask_ids == k).sum().item()
                    original_area = (cur_masks[k] >= 0.5).sum().item()
                    mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        if mask_area / original_area < self.overlap_threshold:
                            continue

                        # merge stuff regions
                        if not isthing:
                            if int(pred_class) in stuff_memory_list.keys():
                                panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                                continue
                            else:
                                stuff_memory_list[int(pred_class)] = current_segment_id + 1

                        current_segment_id += 1
                        panoptic_seg[mask] = current_segment_id

                        segments_info.append(
                            {
                                "id": current_segment_id,
                                "isthing": bool(isthing),
                                "category_id": int(pred_class),
                            }
                        )

                return panoptic_seg, segments_info

        def instance_inference(self, mask_cls, mask_pred):
            # mask_pred is already processed to have the same shape as original input
            image_size = mask_pred.shape[-2:]

            # [Q, K]
            # scores = F.softmax(mask_cls, dim=-1)[:, :-1]
            scores = F.softmax(mask_cls, dim=-1)[:, 1:]
            num_queries, num_classes = scores.shape
            labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
            # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
            labels_per_image = labels[topk_indices]

            topk_indices = topk_indices // num_classes
            # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
            mask_pred = mask_pred[topk_indices]

            # if this is panoptic segmentation, we only keep the "thing" classes
            if self.panoptic_on:
                keep = torch.zeros_like(scores_per_image).bool()
                for i, lab in enumerate(labels_per_image):
                    keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]
                mask_pred = mask_pred[keep]

            result = Instances(image_size)
            # mask (before sigmoid)
            result.pred_masks = (mask_pred > 0).float()
            result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
            # Uncomment the following to get boxes from masks (this is slow)
            # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

            # calculate average mask prob
            mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
                    result.pred_masks.flatten(1).sum(1) + 1e-6)
            result.scores = scores_per_image * mask_scores_per_image
            result.pred_classes = labels_per_image
            return result

elif version == 'posonly':
    class SoftDiceLoss(nn.Module):
        def __init__(self, weight=None, size_average=True):
            super(SoftDiceLoss, self).__init__()

        def forward(self, logits, targets):
            bs = targets.size(0)
            smooth = 1e-6

            probs = F.sigmoid(logits)
            m1 = probs.view(bs, -1)
            m2 = targets.view(bs, -1)
            intersection = (m1 * m2)

            score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
            score = 1 - score.sum() / bs
            return score
    @META_ARCH_REGISTRY.register()
    class IncrementalMaskFormer(nn.Module):
        """
        Main class for mask classification semantic segmentation architectures.
        """

        @configurable
        def __init__(
                self,
                *,
                cfg: None,
                name: str,
                backbone: Backbone,
                sem_seg_head: nn.Module,
                criterion: nn.Module,
                num_queries: int,
                object_mask_threshold: float,
                overlap_threshold: float,
                metadata,
                size_divisibility: int,
                sem_seg_postprocess_before_inference: bool,
                pixel_mean: Tuple[float],
                pixel_std: Tuple[float],
                # inference
                semantic_on: bool,
                panoptic_on: bool,
                instance_on: bool,
                test_topk_per_image: int,
        ):
            """
            Args:
                backbone: a backbone module, must follow detectron2's backbone interface
                sem_seg_head: a module that predicts semantic segmentation from backbone features
                criterion: a module that defines the loss
                num_queries: int, number of queries
                object_mask_threshold: float, threshold to filter query based on classification score
                    for panoptic segmentation inference
                overlap_threshold: overlap threshold used in general inference for panoptic segmentation
                metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                    segmentation inference
                size_divisibility: Some backbones require the input height and width to be divisible by a
                    specific integer. We can use this to override such requirement.
                sem_seg_postprocess_before_inference: whether to resize the prediction back
                    to original input size before semantic segmentation inference or after.
                    For high-resolution dataset like Mapillary, resizing predictions before
                    inference will cause OOM error.
                pixel_mean, pixel_std: list or tuple with #channels element, representing
                    the per-channel mean and std to be used to normalize the input image
                semantic_on: bool, whether to output semantic segmentation prediction
                instance_on: bool, whether to output instance segmentation prediction
                panoptic_on: bool, whether to output panoptic segmentation prediction
                test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
            """
            super().__init__()
            self.cfg = cfg
            dataset, _, task, step = name.split('_')
            self.dataset = dataset
            self.task = task
            self.step = int(step)

            sem_seg_head.predictor.task = self.task
            sem_seg_head.predictor.step = self.step

            self.backbone = backbone
            self.sem_seg_head = sem_seg_head
            self.criterion = criterion
            self.num_queries = num_queries
            self.overlap_threshold = overlap_threshold
            self.object_mask_threshold = object_mask_threshold
            self.metadata = metadata
            if size_divisibility < 0:
                # use backbone size_divisibility if not set
                size_divisibility = self.backbone.size_divisibility
            self.size_divisibility = size_divisibility
            self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
            self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

            # additional args
            self.semantic_on = semantic_on
            self.instance_on = instance_on
            self.panoptic_on = panoptic_on
            self.test_topk_per_image = test_topk_per_image

            if not self.semantic_on:
                assert self.sem_seg_postprocess_before_inference

            self.dice_loss = SoftDiceLoss()

            # incremental setting
            trainable_params = []
            if self.step != 0:
                for name, param in self.named_parameters():
                    if 'incre' not in name:
                        param.requires_grad = True
                    else:
                        trainable_params.append(name)
                print('#' * 50)
                print('Trainable params :')
                print(trainable_params)
                print('#' * 50)

        @classmethod
        def from_config(cls, cfg):
            backbone = build_backbone(cfg)
            sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

            # Loss parameters:
            deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
            no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

            # loss weights
            class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
            dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
            mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

            # building criterion
            matcher = HungarianMatcher(
                cost_class=class_weight,
                cost_mask=mask_weight,
                cost_dice=dice_weight,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            )

            weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

            if deep_supervision:
                dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
                aux_weight_dict = {}
                for i in range(dec_layers - 1):
                    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
                weight_dict.update(aux_weight_dict)

            losses = ["labels", "masks"]

            dataset, seg, task, step = cfg.DATASETS.TRAIN[0].split('_')

            if dataset == 'ade20k':
                if seg == 'incrementalins':
                    from mask2former.data.datasets.register_ade20k_incremental_instance import tasks_ade_ins
                    training_classes = tasks_ade_ins[task][int(step)]
                else:
                    from mask2former.data.datasets.register_ade20k_incremental import tasks_ade
                    training_classes = tasks_ade[task][int(step)]
                cr_num_classes = len(training_classes)
            elif dataset == 'voc':
                from mask2former.data.datasets.register_voc_incremental import tasks_voc
                training_classes = tasks_voc[task][int(step)]
                cr_num_classes = len(training_classes)
            elif dataset == 'coco':
                from mask2former.data.datasets.register_coco_incremental_instance import tasks_coco_ins
                training_classes = tasks_coco_ins[task][int(step)]
                cr_num_classes = len(training_classes)
            else:
                print(f'no dataset {dataset}')
                raise ValueError('No dataset')

            criterion = SetCriterion(
                cr_num_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight,
                losses=losses,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
                oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
                importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            )

            return {
                "cfg": cfg,
                "name": cfg.DATASETS.TRAIN[0],
                "backbone": backbone,
                "sem_seg_head": sem_seg_head,
                "criterion": criterion,
                "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
                "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
                "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
                "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
                "sem_seg_postprocess_before_inference": (
                        cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                        or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                        or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
                ),
                "pixel_mean": cfg.MODEL.PIXEL_MEAN,
                "pixel_std": cfg.MODEL.PIXEL_STD,
                # inference
                "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
                "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
                "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
                "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            }

        @property
        def device(self):
            return self.pixel_mean.device

        def losskd(self,pkd_features,kd_features):
            losskd = dict()
            predictions_classs = kd_features['predictions_class']
            ppredictions_classs = pkd_features['predictions_class']
            predictions_masks = kd_features['predictions_mask']
            ppredictions_masks = pkd_features['predictions_mask']
            # nl = list(range(len(predictions_classs)))
            nl = [9]
            nk = predictions_classs[0].keys()
            for l in nl:
                for k in nk:
                    b, q, c = ppredictions_classs[l][k].shape
                    pred = ppredictions_classs[l][k].reshape(-1,c).softmax(-1)
                    gt = predictions_classs[l][k].reshape(-1,c).softmax(-1)
                    losskd[f'loss_kd_class{l}_{k}'] = F.kl_div(pred.log(),gt,reduction='sum')/(b*q)*100

                    pos = ppredictions_classs[l][k].reshape(-1,c).argmax(-1)>0

                    if pos.sum()<1:
                        losskd[f'loss_kd_mask{l}_{k}'] = pred.sum()*0
                        continue

                    b, q, h, w = predictions_masks[l][k].shape
                    pred = ppredictions_masks[l][k].reshape(-1, h, w)[pos]
                    gt = (predictions_masks[l][k].reshape(-1, h, w) > 0).to(torch.int64)[pos]
                    losskd[f'loss_kd_mask{l}_{k}'] = self.dice_loss(pred, gt) * 100


            return losskd

        def forward(self, batched_inputs, kd=False):
            """
            Args:
                batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                    Each item in the list contains the inputs for one image.
                    For now, each item in the list is a dict that contains:
                       * "image": Tensor, image in (C, H, W) format.
                       * "instances": per-region ground truth
                       * Other information that's included in the original dicts, such as:
                         "height", "width" (int): the output resolution of the model (may be different
                         from input resolution), used in inference.
            Returns:
                list[dict]:
                    each dict has the results for one image. The dict contains the following keys:

                    * "sem_seg":
                        A Tensor that represents the
                        per-pixel segmentation prediced by the head.
                        The prediction has shape KxHxW that represents the logits of
                        each class for each pixel.
                    * "panoptic_seg":
                        A tuple that represent panoptic output
                        panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                        segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                            Each dict contains keys "id", "category_id", "isthing".
            """
            # if False:
            if self.training and not hasattr(self,'teacher_model') and self.step!=0:
                # build teacher model
                teacher_model = build_model(self.cfg)
                teacher_model.step = self.step-1
                teacher_model.sem_seg_head.predictor.step = self.step - 1
                DetectionCheckpointer(teacher_model, save_dir='./cache/debug').resume_or_load(
                    self.cfg.MODEL.WEIGHTS
                )
                teacher_model.eval()
                self.teacher_model = teacher_model
            if self.training and hasattr(self,'teacher_model'):
                kd_features = self.teacher_model(batched_inputs,kd=True)
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            features = self.backbone(images.tensor)

            outputs = self.sem_seg_head(features)

            if kd:
                kd_features = outputs['kd_features']
                kd_features['res_features'] = features
                return kd_features

            if self.training:
                # mask classification target
                if "instances" in batched_inputs[0]:
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                    gt_ignores = [x["ignore"].to(self.device) for x in batched_inputs]
                    targets = self.prepare_targets(gt_instances, gt_ignores, images)
                else:
                    targets = None

                losses = dict()
                if hasattr(self,'teacher_model'):
                    pkd_features = outputs['kd_features']
                    pkd_features['res_features'] = features
                    losses.update(self.losskd(pkd_features, kd_features))
                if 'kd_features' in outputs:
                    outputs.pop('kd_features')
                # bipartite matching-based loss
                losses.update(self.criterion(outputs, targets))

                for k in list(losses.keys()):
                    if k in self.criterion.weight_dict:
                        losses[k] *= self.criterion.weight_dict[k]
                    else:
                        # remove this loss if not specified in `weight_dict`
                        continue
                        # losses.pop(k)
                return losses
            else:
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )

                del outputs

                processed_results = []
                for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                        mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    processed_results.append({})

                    if self.sem_seg_postprocess_before_inference:
                        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            mask_pred_result, image_size, height, width
                        )
                        mask_cls_result = mask_cls_result.to(mask_pred_result)

                    # semantic segmentation inference
                    if self.semantic_on:
                        r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                        if not self.sem_seg_postprocess_before_inference:
                            r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                        processed_results[-1]["sem_seg"] = r

                    # panoptic segmentation inference
                    if self.panoptic_on:
                        panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["panoptic_seg"] = panoptic_r

                    # instance segmentation inference
                    if self.instance_on:
                        instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["instances"] = instance_r

                return processed_results

        def prepare_targets(self, targets, gt_ignores, images):
            h_pad, w_pad = images.tensor.shape[-2:]
            new_targets = []
            for targets_per_image, gt_ignore in zip(targets, gt_ignores):
                # pad gt
                gt_masks = targets_per_image.gt_masks
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
                new_targets.append(
                    {
                        "labels": targets_per_image.gt_classes,
                        "masks": padded_masks,
                        "gt_ignore": gt_ignore,
                    }
                )
            return new_targets

        def semantic_inference(self, mask_cls, mask_pred):
            mask_cls = F.softmax(mask_cls, dim=-1)[..., 1:]  # the first class is the background
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg

        def panoptic_inference(self, mask_cls, mask_pred):
            scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
            mask_pred = mask_pred.sigmoid()

            keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[keep]
            cur_mask_cls = mask_cls[keep]
            cur_mask_cls = cur_mask_cls[:, :-1]

            cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

            h, w = cur_masks.shape[-2:]
            panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
            segments_info = []

            current_segment_id = 0

            if cur_masks.shape[0] == 0:
                # We didn't detect any mask :(
                return panoptic_seg, segments_info
            else:
                # take argmax
                cur_mask_ids = cur_prob_masks.argmax(0)
                stuff_memory_list = {}
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()
                    isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                    mask_area = (cur_mask_ids == k).sum().item()
                    original_area = (cur_masks[k] >= 0.5).sum().item()
                    mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        if mask_area / original_area < self.overlap_threshold:
                            continue

                        # merge stuff regions
                        if not isthing:
                            if int(pred_class) in stuff_memory_list.keys():
                                panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                                continue
                            else:
                                stuff_memory_list[int(pred_class)] = current_segment_id + 1

                        current_segment_id += 1
                        panoptic_seg[mask] = current_segment_id

                        segments_info.append(
                            {
                                "id": current_segment_id,
                                "isthing": bool(isthing),
                                "category_id": int(pred_class),
                            }
                        )

                return panoptic_seg, segments_info

        def instance_inference(self, mask_cls, mask_pred):
            # mask_pred is already processed to have the same shape as original input
            image_size = mask_pred.shape[-2:]

            # [Q, K]
            # scores = F.softmax(mask_cls, dim=-1)[:, :-1]
            scores = F.softmax(mask_cls, dim=-1)[:, 1:]
            num_queries, num_classes = scores.shape
            labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
            # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
            labels_per_image = labels[topk_indices]

            topk_indices = topk_indices // num_classes
            # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
            mask_pred = mask_pred[topk_indices]

            # if this is panoptic segmentation, we only keep the "thing" classes
            if self.panoptic_on:
                keep = torch.zeros_like(scores_per_image).bool()
                for i, lab in enumerate(labels_per_image):
                    keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]
                mask_pred = mask_pred[keep]

            result = Instances(image_size)
            # mask (before sigmoid)
            result.pred_masks = (mask_pred > 0).float()
            result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
            # Uncomment the following to get boxes from masks (this is slow)
            # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

            # calculate average mask prob
            mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
                        result.pred_masks.flatten(1).sum(1) + 1e-6)
            result.scores = scores_per_image * mask_scores_per_image
            result.pred_classes = labels_per_image
            return result

elif version == 'final':
    class SoftDiceLoss(nn.Module):
        def __init__(self, weight=None, size_average=True):
            super(SoftDiceLoss, self).__init__()

        def forward(self, logits, targets):
            bs = targets.size(0)
            smooth = 1.0

            probs = F.sigmoid(logits)
            m1 = probs.view(bs, -1)
            m2 = targets.view(bs, -1)
            intersection = (m1 * m2)

            score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
            score = 1 - score.sum() / bs
            return score

    @META_ARCH_REGISTRY.register()
    class IncrementalMaskFormer(nn.Module):
        """
        Main class for mask classification semantic segmentation architectures.
        """

        @configurable
        def __init__(
                self,
                *,
                cfg: None,
                name: str,
                backbone: Backbone,
                sem_seg_head: nn.Module,
                criterion: nn.Module,
                num_queries: int,
                object_mask_threshold: float,
                overlap_threshold: float,
                metadata,
                size_divisibility: int,
                sem_seg_postprocess_before_inference: bool,
                pixel_mean: Tuple[float],
                pixel_std: Tuple[float],
                # inference
                semantic_on: bool,
                panoptic_on: bool,
                instance_on: bool,
                test_topk_per_image: int,
        ):
            """
            Args:
                backbone: a backbone module, must follow detectron2's backbone interface
                sem_seg_head: a module that predicts semantic segmentation from backbone features
                criterion: a module that defines the loss
                num_queries: int, number of queries
                object_mask_threshold: float, threshold to filter query based on classification score
                    for panoptic segmentation inference
                overlap_threshold: overlap threshold used in general inference for panoptic segmentation
                metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                    segmentation inference
                size_divisibility: Some backbones require the input height and width to be divisible by a
                    specific integer. We can use this to override such requirement.
                sem_seg_postprocess_before_inference: whether to resize the prediction back
                    to original input size before semantic segmentation inference or after.
                    For high-resolution dataset like Mapillary, resizing predictions before
                    inference will cause OOM error.
                pixel_mean, pixel_std: list or tuple with #channels element, representing
                    the per-channel mean and std to be used to normalize the input image
                semantic_on: bool, whether to output semantic segmentation prediction
                instance_on: bool, whether to output instance segmentation prediction
                panoptic_on: bool, whether to output panoptic segmentation prediction
                test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
            """
            super().__init__()
            self.cfg = cfg
            dataset, _, task, step = name.split('_')
            self.dataset = dataset
            self.task = task
            self.step = int(step)

            sem_seg_head.predictor.task = self.task
            sem_seg_head.predictor.step = self.step

            # if self.step>0:
            #     sem_seg_head.predictor.query_feat.weight.requires_grad = False
            #     for s in range(self.step-1):
            #         sem_seg_head.predictor.incre_query_feat_list[s].weight.requires_grad = False

            self.backbone = backbone
            self.sem_seg_head = sem_seg_head
            self.criterion = criterion
            self.num_queries = num_queries
            self.overlap_threshold = overlap_threshold
            self.object_mask_threshold = object_mask_threshold
            self.metadata = metadata
            if size_divisibility < 0:
                # use backbone size_divisibility if not set
                size_divisibility = self.backbone.size_divisibility
            self.size_divisibility = size_divisibility
            self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
            self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

            # additional args
            self.semantic_on = semantic_on
            self.instance_on = instance_on
            self.panoptic_on = panoptic_on
            self.test_topk_per_image = test_topk_per_image

            if not self.semantic_on:
                assert self.sem_seg_postprocess_before_inference

            self.dice_loss = SoftDiceLoss()

        @classmethod
        def from_config(cls, cfg):
            backbone = build_backbone(cfg)
            sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

            # Loss parameters:
            deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
            no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

            # loss weights
            class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
            dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
            mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

            # building criterion
            matcher = HungarianMatcher(
                cost_class=class_weight,
                cost_mask=mask_weight,
                cost_dice=dice_weight,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            )

            weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

            if deep_supervision:
                dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
                aux_weight_dict = {}
                for i in range(dec_layers - 1):
                    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
                weight_dict.update(aux_weight_dict)

            losses = ["labels", "masks"]

            dataset, seg, task, step = cfg.DATASETS.TRAIN[0].split('_')

            if dataset == 'ade20k':
                if seg == 'incrementalins':
                    from mask2former.data.datasets.register_ade20k_incremental_instance import tasks_ade_ins
                    training_classes = tasks_ade_ins[task][int(step)]
                else:
                    from mask2former.data.datasets.register_ade20k_incremental import tasks_ade
                    training_classes = tasks_ade[task][int(step)]
                cr_num_classes = len(training_classes)
            elif dataset == 'voc':
                from mask2former.data.datasets.register_voc_incremental import tasks_voc
                training_classes = tasks_voc[task][int(step)]
                cr_num_classes = len(training_classes)
            elif dataset == 'coco':
                from mask2former.data.datasets.register_coco_incremental_instance import tasks_coco_ins
                training_classes = tasks_coco_ins[task][int(step)]
                cr_num_classes = len(training_classes)
            else:
                print(f'no dataset {dataset}')
                raise ValueError('No dataset')

            criterion = SetCriterion(
                cr_num_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight,
                losses=losses,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
                oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
                importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            )

            return {
                "cfg": cfg,
                "name": cfg.DATASETS.TRAIN[0],
                "backbone": backbone,
                "sem_seg_head": sem_seg_head,
                "criterion": criterion,
                "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
                "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
                "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
                "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
                "sem_seg_postprocess_before_inference": (
                        cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                        or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                        or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
                ),
                "pixel_mean": cfg.MODEL.PIXEL_MEAN,
                "pixel_std": cfg.MODEL.PIXEL_STD,
                # inference
                "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
                "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
                "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
                "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            }

        @property
        def device(self):
            return self.pixel_mean.device

        def losskd_feature(self, pred_feature, gt_feature):
            return (((pred_feature-gt_feature)**2)/(gt_feature**2+1e-6)).mean()

        def losskd(self,pkd_features,kd_features):
            losskd = dict()

            # Prediction KD
            predictions_classs = kd_features['predictions_class']
            ppredictions_classs = pkd_features['predictions_class']
            predictions_masks = kd_features['predictions_mask']
            ppredictions_masks = pkd_features['predictions_mask']
            nl = [9]
            nk = predictions_classs[0].keys()
            for l in nl:
                total_nq = sum([ppredictions_classs[l][k].shape[1] for k in nk])
                losskd[f'loss_kd_class{l}'] = 0
                losskd[f'loss_kd_mask{l}'] = 0
                for k in nk:
                    pred_class = ppredictions_classs[l][k]
                    gt_class = predictions_classs[l][k]

                    b, nq, c = pred_class.shape

                    pred = pred_class.reshape(-1,c).softmax(-1)
                    gt = gt_class.reshape(-1,c).softmax(-1)
                    step_loss_kd_class = (F.kl_div(pred.log(),gt,reduction='sum')/(b*nq)*100)*nq/total_nq
                    losskd[f'loss_kd_class{l}'] += step_loss_kd_class

                    pos = pred_class.reshape(-1,c).argmax(-1)>0

                    if pos.sum()<1:
                        losskd[f'loss_kd_mask{l}'] += pred.sum()*0
                        continue

                    pred_mask = ppredictions_masks[l][k]
                    gt_mask = predictions_masks[l][k]

                    b, nq, h, w = pred_mask.shape
                    pred = pred_mask.reshape(-1, h, w)
                    gt = (gt_mask.reshape(-1, h, w) > 0).to(torch.int64)
                    step_loss_kd_mask = (self.dice_loss(pred, gt) * 100 + 50.)*nq/total_nq
                    losskd[f'loss_kd_mask{l}'] += step_loss_kd_mask

                # Query KD
                predictions_query = kd_features['predictions_query']
                ppredictions_query = pkd_features['predictions_query']
                nq_previous = len(predictions_query[0])
                loss_qkd = 0
                for query, pred_query in zip(predictions_query, ppredictions_query):
                    loss_qkd = (torch.abs(query-pred_query[:nq_previous])/(torch.abs(query)+1e-6)).mean()
                loss_qkd = loss_qkd/len(predictions_query)
                losskd[f'loss_kd_query'] = loss_qkd * 10

            # losskd[f'loss_kd_maskfeatures'] = self.losskd_feature(pkd_features['mask_features'], kd_features['mask_features'])*10


            # keys = list(pkd_features['res_features'].keys())
            # for key in keys:
            #     losskd[f'loss_kd_{key}'] = self.losskd_feature(pkd_features['res_features'][key], kd_features['res_features'][key])*10


            return losskd

        def detach_features(self, features):
            if isinstance(features,torch.Tensor):
                return features.detach()
            elif isinstance(features,dict):
                new_dict = dict()
                for k,v in features.items():
                    new_dict[k] = self.detach_features(v)
                return new_dict
            elif isinstance(features,list):
                return [self.detach_features(el) for el in features]
            else:
                raise ValueError("unknow type")


        def forward(self, batched_inputs, kd=False):
            """
            Args:
                batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                    Each item in the list contains the inputs for one image.
                    For now, each item in the list is a dict that contains:
                       * "image": Tensor, image in (C, H, W) format.
                       * "instances": per-region ground truth
                       * Other information that's included in the original dicts, such as:
                         "height", "width" (int): the output resolution of the model (may be different
                         from input resolution), used in inference.
            Returns:
                list[dict]:
                    each dict has the results for one image. The dict contains the following keys:

                    * "sem_seg":
                        A Tensor that represents the
                        per-pixel segmentation prediced by the head.
                        The prediction has shape KxHxW that represents the logits of
                        each class for each pixel.
                    * "panoptic_seg":
                        A tuple that represent panoptic output
                        panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                        segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                            Each dict contains keys "id", "category_id", "isthing".
            """
            # get the teacher model if current step is not base
            if self.training and not hasattr(self,'teacher_model') and self.step>0:
                # build teacher model
                teacher_model = build_model(self.cfg)
                teacher_model.step = self.step-1
                teacher_model.sem_seg_head.predictor.step = self.step - 1
                DetectionCheckpointer(teacher_model, save_dir='./cache/debug').resume_or_load(
                    self.cfg.MODEL.WEIGHTS
                )
                teacher_model.eval()
                self.teacher_model = teacher_model
            if self.training and hasattr(self,'teacher_model'):
                kd_features = self.teacher_model(batched_inputs,kd=True)


            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            features = self.backbone(images.tensor)

            outputs = self.sem_seg_head(features)

            if kd:
                kd_features = outputs['kd_features']
                kd_features['res_features'] = features
                return self.detach_features(kd_features)

            if self.training:
                # mask classification target
                if "instances" in batched_inputs[0]:
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                    gt_ignores = [x["ignore"].to(self.device) for x in batched_inputs]
                    targets = self.prepare_targets(gt_instances, gt_ignores, images)
                else:
                    targets = None

                losses = dict()
                if hasattr(self,'teacher_model'):
                    pkd_features = outputs['kd_features']
                    pkd_features['res_features'] = features
                    losses.update(self.losskd(pkd_features, kd_features))
                if 'kd_features' in outputs:
                    outputs.pop('kd_features')
                # bipartite matching-based loss
                losses.update(self.criterion(outputs, targets))

                for k in list(losses.keys()):
                    if k in self.criterion.weight_dict:
                        losses[k] *= self.criterion.weight_dict[k]
                    else:
                        # remove this loss if not specified in `weight_dict`
                        continue
                        # losses.pop(k)
                return losses
            else:
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )

                del outputs

                processed_results = []
                for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                        mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    processed_results.append({})

                    if self.sem_seg_postprocess_before_inference:
                        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            mask_pred_result, image_size, height, width
                        )
                        mask_cls_result = mask_cls_result.to(mask_pred_result)

                    # semantic segmentation inference
                    if self.semantic_on:
                        r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                        if not self.sem_seg_postprocess_before_inference:
                            r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                        processed_results[-1]["sem_seg"] = r

                    # panoptic segmentation inference
                    if self.panoptic_on:
                        panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["panoptic_seg"] = panoptic_r

                    # instance segmentation inference
                    if self.instance_on:
                        instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["instances"] = instance_r

                return processed_results

        def prepare_targets(self, targets, gt_ignores, images):
            h_pad, w_pad = images.tensor.shape[-2:]
            new_targets = []
            for targets_per_image, gt_ignore in zip(targets, gt_ignores):
                # pad gt
                gt_masks = targets_per_image.gt_masks
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
                new_targets.append(
                    {
                        "labels": targets_per_image.gt_classes,
                        "masks": padded_masks,
                        "gt_ignore": gt_ignore,
                    }
                )
            return new_targets

        def semantic_inference(self, mask_cls, mask_pred):

            mask_cls = F.softmax(mask_cls, dim=-1)[..., 1:]  # the first class is the background
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg

        def panoptic_inference(self, mask_cls, mask_pred):
            scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
            mask_pred = mask_pred.sigmoid()

            keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[keep]
            cur_mask_cls = mask_cls[keep]
            cur_mask_cls = cur_mask_cls[:, :-1]

            cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

            h, w = cur_masks.shape[-2:]
            panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
            segments_info = []

            current_segment_id = 0

            if cur_masks.shape[0] == 0:
                # We didn't detect any mask :(
                return panoptic_seg, segments_info
            else:
                # take argmax
                cur_mask_ids = cur_prob_masks.argmax(0)
                stuff_memory_list = {}
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()
                    isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                    mask_area = (cur_mask_ids == k).sum().item()
                    original_area = (cur_masks[k] >= 0.5).sum().item()
                    mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        if mask_area / original_area < self.overlap_threshold:
                            continue

                        # merge stuff regions
                        if not isthing:
                            if int(pred_class) in stuff_memory_list.keys():
                                panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                                continue
                            else:
                                stuff_memory_list[int(pred_class)] = current_segment_id + 1

                        current_segment_id += 1
                        panoptic_seg[mask] = current_segment_id

                        segments_info.append(
                            {
                                "id": current_segment_id,
                                "isthing": bool(isthing),
                                "category_id": int(pred_class),
                            }
                        )

                return panoptic_seg, segments_info

        def instance_inference(self, mask_cls, mask_pred):
            # mask_pred is already processed to have the same shape as original input
            image_size = mask_pred.shape[-2:]

            # [Q, K]
            # scores = F.softmax(mask_cls, dim=-1)[:, :-1]
            scores = F.softmax(mask_cls, dim=-1)[:, 1:]
            num_queries, num_classes = scores.shape
            labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
            # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
            labels_per_image = labels[topk_indices]

            topk_indices = topk_indices // num_classes
            # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
            mask_pred = mask_pred[topk_indices]

            # if this is panoptic segmentation, we only keep the "thing" classes
            if self.panoptic_on:
                keep = torch.zeros_like(scores_per_image).bool()
                for i, lab in enumerate(labels_per_image):
                    keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]
                mask_pred = mask_pred[keep]

            result = Instances(image_size)
            # mask (before sigmoid)
            result.pred_masks = (mask_pred > 0).float()
            result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
            # Uncomment the following to get boxes from masks (this is slow)
            # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

            # calculate average mask prob
            mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
                        result.pred_masks.flatten(1).sum(1) + 1e-6)
            result.scores = scores_per_image * mask_scores_per_image
            result.pred_classes = labels_per_image
            return result

elif version == 'final2':
    # query freeze : True
    # query kd : True
    # prediction kd : True
    # query group attention mask : True
    from incremental_utils import features_distillation
    class SoftDiceLoss(nn.Module):
        def __init__(self, weight=None, size_average=True):
            super(SoftDiceLoss, self).__init__()

        def forward(self, logits, targets):
            bs = targets.size(0)
            smooth = 1.0

            probs = F.sigmoid(logits)
            m1 = probs.view(bs, -1)
            m2 = targets.view(bs, -1)
            intersection = (m1 * m2)

            score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
            score = 1 - score.sum() / bs
            return score

    @META_ARCH_REGISTRY.register()
    class IncrementalMaskFormer(nn.Module):
        """
        Main class for mask classification semantic segmentation architectures.
        """

        @configurable
        def __init__(
                self,
                *,
                cfg: None,
                name: str,
                backbone: Backbone,
                sem_seg_head: nn.Module,
                criterion: nn.Module,
                num_queries: int,
                object_mask_threshold: float,
                overlap_threshold: float,
                metadata,
                size_divisibility: int,
                sem_seg_postprocess_before_inference: bool,
                pixel_mean: Tuple[float],
                pixel_std: Tuple[float],
                # inference
                semantic_on: bool,
                panoptic_on: bool,
                instance_on: bool,
                test_topk_per_image: int,
        ):
            """
            Args:
                backbone: a backbone module, must follow detectron2's backbone interface
                sem_seg_head: a module that predicts semantic segmentation from backbone features
                criterion: a module that defines the loss
                num_queries: int, number of queries
                object_mask_threshold: float, threshold to filter query based on classification score
                    for panoptic segmentation inference
                overlap_threshold: overlap threshold used in general inference for panoptic segmentation
                metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                    segmentation inference
                size_divisibility: Some backbones require the input height and width to be divisible by a
                    specific integer. We can use this to override such requirement.
                sem_seg_postprocess_before_inference: whether to resize the prediction back
                    to original input size before semantic segmentation inference or after.
                    For high-resolution dataset like Mapillary, resizing predictions before
                    inference will cause OOM error.
                pixel_mean, pixel_std: list or tuple with #channels element, representing
                    the per-channel mean and std to be used to normalize the input image
                semantic_on: bool, whether to output semantic segmentation prediction
                instance_on: bool, whether to output instance segmentation prediction
                panoptic_on: bool, whether to output panoptic segmentation prediction
                test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
            """
            super().__init__()
            self.cfg = cfg
            dataset, _, task, step = name.split('_')
            self.dataset = dataset
            self.task = task
            self.step = int(step)

            sem_seg_head.predictor.task = self.task
            sem_seg_head.predictor.step = self.step

            if self.step>0:
                sem_seg_head.predictor.query_feat.weight.requires_grad = False
                for s in range(self.step-1):
                    sem_seg_head.predictor.incre_query_feat_list[s].weight.requires_grad = False

            self.backbone = backbone
            self.sem_seg_head = sem_seg_head
            self.criterion = criterion
            self.num_queries = num_queries
            self.overlap_threshold = overlap_threshold
            self.object_mask_threshold = object_mask_threshold
            self.metadata = metadata
            if size_divisibility < 0:
                # use backbone size_divisibility if not set
                size_divisibility = self.backbone.size_divisibility
            self.size_divisibility = size_divisibility
            self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
            self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

            # additional args
            self.semantic_on = semantic_on
            self.instance_on = instance_on
            self.panoptic_on = panoptic_on
            self.test_topk_per_image = test_topk_per_image

            if not self.semantic_on:
                assert self.sem_seg_postprocess_before_inference

            self.dice_loss = SoftDiceLoss()

        @classmethod
        def from_config(cls, cfg):
            backbone = build_backbone(cfg)
            sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

            # Loss parameters:
            deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
            no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

            # loss weights
            class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
            dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
            mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

            # building criterion
            matcher = HungarianMatcher(
                cost_class=class_weight,
                cost_mask=mask_weight,
                cost_dice=dice_weight,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            )

            weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

            if deep_supervision:
                dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
                aux_weight_dict = {}
                for i in range(dec_layers - 1):
                    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
                weight_dict.update(aux_weight_dict)

            losses = ["labels", "masks"]

            dataset, seg, task, step = cfg.DATASETS.TRAIN[0].split('_')

            if dataset == 'ade20k':
                if seg == 'incrementalins':
                    from mask2former.data.datasets.register_ade20k_incremental_instance import tasks_ade_ins
                    training_classes = tasks_ade_ins[task][int(step)]
                else:
                    from mask2former.data.datasets.register_ade20k_incremental import tasks_ade
                    training_classes = tasks_ade[task][int(step)]
                cr_num_classes = len(training_classes)
            elif dataset == 'voc':
                from mask2former.data.datasets.register_voc_incremental import tasks_voc
                training_classes = tasks_voc[task][int(step)]
                cr_num_classes = len(training_classes)
            elif dataset == 'coco':
                from mask2former.data.datasets.register_coco_incremental_instance import tasks_coco_ins
                training_classes = tasks_coco_ins[task][int(step)]
                cr_num_classes = len(training_classes)
            else:
                print(f'no dataset {dataset}')
                raise ValueError('No dataset')

            criterion = SetCriterion(
                cr_num_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight,
                losses=losses,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
                oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
                importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            )

            return {
                "cfg": cfg,
                "name": cfg.DATASETS.TRAIN[0],
                "backbone": backbone,
                "sem_seg_head": sem_seg_head,
                "criterion": criterion,
                "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
                "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
                "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
                "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
                "sem_seg_postprocess_before_inference": (
                        cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                        or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                        or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
                ),
                "pixel_mean": cfg.MODEL.PIXEL_MEAN,
                "pixel_std": cfg.MODEL.PIXEL_STD,
                # inference
                "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
                "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
                "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
                "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            }

        @property
        def device(self):
            return self.pixel_mean.device

        def losskd_feature(self, pred_feature, gt_feature):
            return (((pred_feature-gt_feature)**2)/(gt_feature**2+1e-6)).mean()

        def losskd(self,pkd_features,kd_features):
            losskd = dict()

            # Prediction KD
            predictions_classs = kd_features['predictions_class']
            ppredictions_classs = pkd_features['predictions_class']
            predictions_masks = kd_features['predictions_mask']
            ppredictions_masks = pkd_features['predictions_mask']
            nl = [9]
            nk = predictions_classs[0].keys()
            for l in nl:
                total_nq = sum([ppredictions_classs[l][k].shape[1] for k in nk])
                losskd[f'loss_kd_class{l}'] = 0
                losskd[f'loss_kd_mask{l}'] = 0
                for k in nk:
                    pred_class = ppredictions_classs[l][k]
                    gt_class = predictions_classs[l][k]

                    b, nq, c = pred_class.shape

                    pred = pred_class.reshape(-1,c).softmax(-1)
                    gt = gt_class.reshape(-1,c).softmax(-1)
                    # 1000 for 11 task, 300 for 2 or 3 task
                    step_loss_kd_class = (F.kl_div(pred.log(),gt,reduction='sum')/(b*nq)*300)*nq/total_nq
                    losskd[f'loss_kd_class{l}'] += step_loss_kd_class

                    pred_mask = ppredictions_masks[l][k]
                    gt_mask = predictions_masks[l][k]

                    b, nq, h, w = pred_mask.shape
                    pred = pred_mask.reshape(-1, h, w)
                    gt = (gt_mask.reshape(-1, h, w) > 0).to(torch.int64)
                    step_loss_kd_mask = (self.dice_loss(pred, gt) * 300)*nq/total_nq
                    losskd[f'loss_kd_mask{l}'] += step_loss_kd_mask

            # Query KD
            predictions_query = kd_features['predictions_query']
            ppredictions_query = pkd_features['predictions_query']
            nq_previous = len(predictions_query[0])
            loss_qkd = 0
            for query, pred_query in zip(predictions_query, ppredictions_query):
                loss_qkd = (torch.abs(query-pred_query[:nq_previous])/(torch.abs(query)+1e-6)).mean()
            loss_qkd = loss_qkd/len(predictions_query)
            losskd[f'loss_kd_query'] = loss_qkd * 10

            losskd[f'loss_kd_maskfeatures'] = self.losskd_feature(pkd_features['mask_features'], kd_features['mask_features'])*10


            # keys = list(pkd_features['res_features'].keys())
            # for key in keys:
            #     losskd[f'loss_kd_{key}'] = self.losskd_feature(pkd_features['res_features'][key], kd_features['res_features'][key])*10


            return losskd

        def detach_features(self, features):
            if isinstance(features,torch.Tensor):
                return features.detach()
            elif isinstance(features,dict):
                new_dict = dict()
                for k,v in features.items():
                    new_dict[k] = self.detach_features(v)
                return new_dict
            elif isinstance(features,list):
                return [self.detach_features(el) for el in features]
            else:
                raise ValueError("unknow type")


        def forward(self, batched_inputs, kd=False):
            """
            Args:
                batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                    Each item in the list contains the inputs for one image.
                    For now, each item in the list is a dict that contains:
                       * "image": Tensor, image in (C, H, W) format.
                       * "instances": per-region ground truth
                       * Other information that's included in the original dicts, such as:
                         "height", "width" (int): the output resolution of the model (may be different
                         from input resolution), used in inference.
            Returns:
                list[dict]:
                    each dict has the results for one image. The dict contains the following keys:

                    * "sem_seg":
                        A Tensor that represents the
                        per-pixel segmentation prediced by the head.
                        The prediction has shape KxHxW that represents the logits of
                        each class for each pixel.
                    * "panoptic_seg":
                        A tuple that represent panoptic output
                        panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                        segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                            Each dict contains keys "id", "category_id", "isthing".
            """
            # get the teacher model if current step is not base
            if self.training and not hasattr(self,'teacher_model') and self.step>0:
                # build teacher model
                teacher_model = build_model(self.cfg)
                teacher_model.step = self.step-1
                teacher_model.sem_seg_head.predictor.step = self.step - 1
                DetectionCheckpointer(teacher_model, save_dir='./cache/debug').resume_or_load(
                    self.cfg.MODEL.WEIGHTS
                )
                teacher_model.eval()
                self.teacher_model = teacher_model
            if self.training and hasattr(self,'teacher_model'):
                kd_features = self.teacher_model(batched_inputs,kd=True)


            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            features = self.backbone(images.tensor)

            outputs = self.sem_seg_head(features)

            if kd:
                kd_features = outputs['kd_features']
                kd_features['res_features'] = features
                return self.detach_features(kd_features)

            if self.training:
                # mask classification target
                if "instances" in batched_inputs[0]:
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                    gt_ignores = [x["ignore"].to(self.device) for x in batched_inputs]
                    targets = self.prepare_targets(gt_instances, gt_ignores, images)
                else:
                    targets = None

                losses = dict()
                if hasattr(self,'teacher_model'):
                    pkd_features = outputs['kd_features']
                    pkd_features['res_features'] = features
                    losses.update(self.losskd(pkd_features, kd_features))
                if 'kd_features' in outputs:
                    outputs.pop('kd_features')
                # bipartite matching-based loss
                losses.update(self.criterion(outputs, targets))

                for k in list(losses.keys()):
                    if k in self.criterion.weight_dict:
                        losses[k] *= self.criterion.weight_dict[k]
                    else:
                        # remove this loss if not specified in `weight_dict`
                        continue
                        # losses.pop(k)
                return losses
            else:
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )

                del outputs

                processed_results = []
                for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                        mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    processed_results.append({})

                    if self.sem_seg_postprocess_before_inference:
                        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            mask_pred_result, image_size, height, width
                        )
                        mask_cls_result = mask_cls_result.to(mask_pred_result)

                    # semantic segmentation inference
                    if self.semantic_on:
                        r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                        if not self.sem_seg_postprocess_before_inference:
                            r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                        processed_results[-1]["sem_seg"] = r

                    # panoptic segmentation inference
                    if self.panoptic_on:
                        panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["panoptic_seg"] = panoptic_r

                    # instance segmentation inference
                    if self.instance_on:
                        instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["instances"] = instance_r

                return processed_results

        def prepare_targets(self, targets, gt_ignores, images):
            h_pad, w_pad = images.tensor.shape[-2:]
            new_targets = []
            for targets_per_image, gt_ignore in zip(targets, gt_ignores):
                # pad gt
                gt_masks = targets_per_image.gt_masks
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
                new_targets.append(
                    {
                        "labels": targets_per_image.gt_classes,
                        "masks": padded_masks,
                        "gt_ignore": gt_ignore,
                    }
                )
            return new_targets

        def semantic_inference(self, mask_cls, mask_pred):

            mask_cls = F.softmax(mask_cls, dim=-1)[..., 1:]  # the first class is the background
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg

        def panoptic_inference(self, mask_cls, mask_pred):
            scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
            mask_pred = mask_pred.sigmoid()

            keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[keep]
            cur_mask_cls = mask_cls[keep]
            cur_mask_cls = cur_mask_cls[:, :-1]

            cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

            h, w = cur_masks.shape[-2:]
            panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
            segments_info = []

            current_segment_id = 0

            if cur_masks.shape[0] == 0:
                # We didn't detect any mask :(
                return panoptic_seg, segments_info
            else:
                # take argmax
                cur_mask_ids = cur_prob_masks.argmax(0)
                stuff_memory_list = {}
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()
                    isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                    mask_area = (cur_mask_ids == k).sum().item()
                    original_area = (cur_masks[k] >= 0.5).sum().item()
                    mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        if mask_area / original_area < self.overlap_threshold:
                            continue

                        # merge stuff regions
                        if not isthing:
                            if int(pred_class) in stuff_memory_list.keys():
                                panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                                continue
                            else:
                                stuff_memory_list[int(pred_class)] = current_segment_id + 1

                        current_segment_id += 1
                        panoptic_seg[mask] = current_segment_id

                        segments_info.append(
                            {
                                "id": current_segment_id,
                                "isthing": bool(isthing),
                                "category_id": int(pred_class),
                            }
                        )

                return panoptic_seg, segments_info

        def instance_inference(self, mask_cls, mask_pred):
            # mask_pred is already processed to have the same shape as original input
            image_size = mask_pred.shape[-2:]

            # [Q, K]
            # scores = F.softmax(mask_cls, dim=-1)[:, :-1]
            scores = F.softmax(mask_cls, dim=-1)[:, 1:]
            num_queries, num_classes = scores.shape
            labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
            # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
            labels_per_image = labels[topk_indices]

            topk_indices = topk_indices // num_classes
            # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
            mask_pred = mask_pred[topk_indices]

            # if this is panoptic segmentation, we only keep the "thing" classes
            if self.panoptic_on:
                keep = torch.zeros_like(scores_per_image).bool()
                for i, lab in enumerate(labels_per_image):
                    keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]
                mask_pred = mask_pred[keep]

            result = Instances(image_size)
            # mask (before sigmoid)
            result.pred_masks = (mask_pred > 0).float()
            result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
            # Uncomment the following to get boxes from masks (this is slow)
            # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

            # calculate average mask prob
            mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
                        result.pred_masks.flatten(1).sum(1) + 1e-6)
            result.scores = scores_per_image * mask_scores_per_image
            result.pred_classes = labels_per_image
            return result

elif version == 'final3':
    # query freeze : True
    # query kd : True
    # prediction kd : True
    # query group attention mask : True
    class SoftDiceLoss(nn.Module):
        def __init__(self, weight=None, size_average=True):
            super(SoftDiceLoss, self).__init__()

        def forward(self, logits, targets):
            bs = targets.size(0)
            smooth = 1.0

            probs = F.sigmoid(logits)
            m1 = probs.view(bs, -1)
            m2 = targets.view(bs, -1)
            intersection = (m1 * m2)

            score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
            score = 1 - score.sum() / bs
            return score

    @META_ARCH_REGISTRY.register()
    class IncrementalMaskFormer(nn.Module):
        """
        Main class for mask classification semantic segmentation architectures.
        """

        @configurable
        def __init__(
                self,
                *,
                cfg: None,
                name: str,
                backbone: Backbone,
                sem_seg_head: nn.Module,
                criterion: nn.Module,
                num_queries: int,
                object_mask_threshold: float,
                overlap_threshold: float,
                metadata,
                size_divisibility: int,
                sem_seg_postprocess_before_inference: bool,
                pixel_mean: Tuple[float],
                pixel_std: Tuple[float],
                # inference
                semantic_on: bool,
                panoptic_on: bool,
                instance_on: bool,
                test_topk_per_image: int,
        ):
            """
            Args:
                backbone: a backbone module, must follow detectron2's backbone interface
                sem_seg_head: a module that predicts semantic segmentation from backbone features
                criterion: a module that defines the loss
                num_queries: int, number of queries
                object_mask_threshold: float, threshold to filter query based on classification score
                    for panoptic segmentation inference
                overlap_threshold: overlap threshold used in general inference for panoptic segmentation
                metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                    segmentation inference
                size_divisibility: Some backbones require the input height and width to be divisible by a
                    specific integer. We can use this to override such requirement.
                sem_seg_postprocess_before_inference: whether to resize the prediction back
                    to original input size before semantic segmentation inference or after.
                    For high-resolution dataset like Mapillary, resizing predictions before
                    inference will cause OOM error.
                pixel_mean, pixel_std: list or tuple with #channels element, representing
                    the per-channel mean and std to be used to normalize the input image
                semantic_on: bool, whether to output semantic segmentation prediction
                instance_on: bool, whether to output instance segmentation prediction
                panoptic_on: bool, whether to output panoptic segmentation prediction
                test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
            """
            super().__init__()
            self.cfg = cfg
            dataset, _, task, step = name.split('_')
            self.dataset = dataset
            self.task = task
            self.step = int(step)

            sem_seg_head.predictor.task = self.task
            sem_seg_head.predictor.step = self.step

            if self.step>0:
                sem_seg_head.predictor.query_feat.weight.requires_grad = False
                for s in range(self.step-1):
                    sem_seg_head.predictor.incre_query_feat_list[s].weight.requires_grad = False

            self.backbone = backbone
            self.sem_seg_head = sem_seg_head
            self.criterion = criterion
            self.num_queries = num_queries
            self.overlap_threshold = overlap_threshold
            self.object_mask_threshold = object_mask_threshold
            self.metadata = metadata
            if size_divisibility < 0:
                # use backbone size_divisibility if not set
                size_divisibility = self.backbone.size_divisibility
            self.size_divisibility = size_divisibility
            self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
            self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

            # additional args
            self.semantic_on = semantic_on
            self.instance_on = instance_on
            self.panoptic_on = panoptic_on
            self.test_topk_per_image = test_topk_per_image

            if not self.semantic_on:
                assert self.sem_seg_postprocess_before_inference

            self.dice_loss = SoftDiceLoss()

        @classmethod
        def from_config(cls, cfg):
            backbone = build_backbone(cfg)
            sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

            # Loss parameters:
            deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
            no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

            # loss weights
            class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
            dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
            mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

            # building criterion
            matcher = HungarianMatcher(
                cost_class=class_weight,
                cost_mask=mask_weight,
                cost_dice=dice_weight,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            )

            weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

            if deep_supervision:
                dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
                aux_weight_dict = {}
                for i in range(dec_layers - 1):
                    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
                weight_dict.update(aux_weight_dict)

            losses = ["labels", "masks"]

            dataset, seg, task, step = cfg.DATASETS.TRAIN[0].split('_')

            if dataset == 'ade20k':
                if seg == 'incrementalins':
                    from mask2former.data.datasets.register_ade20k_incremental_instance import tasks_ade_ins
                    training_classes = tasks_ade_ins[task][int(step)]
                else:
                    from mask2former.data.datasets.register_ade20k_incremental import tasks_ade
                    training_classes = tasks_ade[task][int(step)]
                cr_num_classes = len(training_classes)
            elif dataset == 'voc':
                from mask2former.data.datasets.register_voc_incremental import tasks_voc
                training_classes = tasks_voc[task][int(step)]
                cr_num_classes = len(training_classes)
            elif dataset == 'coco':
                from mask2former.data.datasets.register_coco_incremental_instance import tasks_coco_ins
                training_classes = tasks_coco_ins[task][int(step)]
                cr_num_classes = len(training_classes)
            else:
                print(f'no dataset {dataset}')
                raise ValueError('No dataset')

            criterion = SetCriterion(
                cr_num_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight,
                losses=losses,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
                oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
                importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            )

            return {
                "cfg": cfg,
                "name": cfg.DATASETS.TRAIN[0],
                "backbone": backbone,
                "sem_seg_head": sem_seg_head,
                "criterion": criterion,
                "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
                "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
                "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
                "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
                "sem_seg_postprocess_before_inference": (
                        cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                        or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                        or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
                ),
                "pixel_mean": cfg.MODEL.PIXEL_MEAN,
                "pixel_std": cfg.MODEL.PIXEL_STD,
                # inference
                "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
                "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
                "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
                "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            }

        @property
        def device(self):
            return self.pixel_mean.device

        def losskd_feature(self, pred_feature, gt_feature):
            return (((pred_feature-gt_feature)**2)/(gt_feature**2+1e-6)).mean()

        def losskd(self,pkd_features,kd_features):
            losskd = dict()

            # Prediction KD
            predictions_classs = kd_features['predictions_class']
            ppredictions_classs = pkd_features['predictions_class']
            predictions_masks = kd_features['predictions_mask']
            ppredictions_masks = pkd_features['predictions_mask']
            nl = [9]
            nk = predictions_classs[0].keys()
            for l in nl:
                total_nq = sum([ppredictions_classs[l][k].shape[1] for k in nk])
                losskd[f'loss_kd_class{l}'] = 0
                losskd[f'loss_kd_mask{l}'] = 0
                for k in nk:
                    pred_class = ppredictions_classs[l][k]
                    gt_class = predictions_classs[l][k]

                    b, nq, c = pred_class.shape

                    pred = pred_class.reshape(-1,c).softmax(-1)
                    gt = gt_class.reshape(-1,c).softmax(-1)
                    # 1000 for 11 task, 300 for 2 or 3 task
                    step_loss_kd_class = (F.kl_div(pred.log(),gt,reduction='sum')/(b*nq)*1000)*nq/total_nq
                    losskd[f'loss_kd_class{l}'] += step_loss_kd_class

                    pred_mask = ppredictions_masks[l][k]
                    gt_mask = predictions_masks[l][k]

                    b, nq, h, w = pred_mask.shape
                    pred = pred_mask.reshape(-1, h, w)
                    gt = (gt_mask.reshape(-1, h, w) > 0).to(torch.int64)
                    step_loss_kd_mask = (self.dice_loss(pred, gt) * 1000)*nq/total_nq
                    losskd[f'loss_kd_mask{l}'] += step_loss_kd_mask

                # Query KD
                predictions_query = kd_features['predictions_query']
                ppredictions_query = pkd_features['predictions_query']
                nq_previous = len(predictions_query[0])
                loss_qkd = 0
                for query, pred_query in zip(predictions_query, ppredictions_query):
                    loss_qkd = (torch.abs(query-pred_query[:nq_previous])/(torch.abs(query)+1e-6)).mean()
                loss_qkd = loss_qkd/len(predictions_query)
                losskd[f'loss_kd_query'] = loss_qkd * 10

            # losskd[f'loss_kd_maskfeatures'] = self.losskd_feature(pkd_features['mask_features'], kd_features['mask_features'])*10


            # keys = list(pkd_features['res_features'].keys())
            # for key in keys:
            #     losskd[f'loss_kd_{key}'] = self.losskd_feature(pkd_features['res_features'][key], kd_features['res_features'][key])*10


            return losskd

        def detach_features(self, features):
            if isinstance(features,torch.Tensor):
                return features.detach()
            elif isinstance(features,dict):
                new_dict = dict()
                for k,v in features.items():
                    new_dict[k] = self.detach_features(v)
                return new_dict
            elif isinstance(features,list):
                return [self.detach_features(el) for el in features]
            else:
                raise ValueError("unknow type")


        def forward(self, batched_inputs, kd=False):
            """
            Args:
                batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                    Each item in the list contains the inputs for one image.
                    For now, each item in the list is a dict that contains:
                       * "image": Tensor, image in (C, H, W) format.
                       * "instances": per-region ground truth
                       * Other information that's included in the original dicts, such as:
                         "height", "width" (int): the output resolution of the model (may be different
                         from input resolution), used in inference.
            Returns:
                list[dict]:
                    each dict has the results for one image. The dict contains the following keys:

                    * "sem_seg":
                        A Tensor that represents the
                        per-pixel segmentation prediced by the head.
                        The prediction has shape KxHxW that represents the logits of
                        each class for each pixel.
                    * "panoptic_seg":
                        A tuple that represent panoptic output
                        panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                        segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                            Each dict contains keys "id", "category_id", "isthing".
            """
            # get the teacher model if current step is not base
            if self.training and not hasattr(self,'teacher_model') and self.step>0:
                novel_teacher_weight = self.cfg.MODEL.WEIGHTS
                if 'step' not in novel_teacher_weight:
                    base_teacher_weight = novel_teacher_weight
                else:
                    base_teacher_weight = novel_teacher_weight.split('/')
                    base_teacher_weight[-2] = 'step0'
                    base_teacher_weight = '/'.join(base_teacher_weight)
                # build teacher model
                nteacher_model = build_model(self.cfg)
                nteacher_model.step = self.step-1
                nteacher_model.sem_seg_head.predictor.step = self.step - 1
                DetectionCheckpointer(nteacher_model, save_dir='./cache/debug').resume_or_load(
                    novel_teacher_weight
                )
                nteacher_model.eval()
                self.nteacher_model = nteacher_model

                bteacher_model = build_model(self.cfg)
                bteacher_model.step = 0
                bteacher_model.sem_seg_head.predictor.step = 0
                DetectionCheckpointer(bteacher_model, save_dir='./cache/debug').resume_or_load(
                    base_teacher_weight
                )
                bteacher_model.eval()
                self.bteacher_model = bteacher_model
            if self.training and hasattr(self,'teacher_model'):
                if self.step==1:
                    kd_features = self.nteacher_model(batched_inputs,kd=True)
                else:
                    nkd_features = self.nteacher_model(batched_inputs,kd=True)
                    bkd_features = self.bteacher_model(batched_inputs,kd=True)


            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            features = self.backbone(images.tensor)

            outputs = self.sem_seg_head(features)

            if kd:
                kd_features = outputs['kd_features']
                kd_features['res_features'] = features
                return self.detach_features(kd_features)

            if self.training:
                # mask classification target
                if "instances" in batched_inputs[0]:
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                    gt_ignores = [x["ignore"].to(self.device) for x in batched_inputs]
                    targets = self.prepare_targets(gt_instances, gt_ignores, images)
                else:
                    targets = None

                losses = dict()
                if hasattr(self,'teacher_model'):
                    pkd_features = outputs['kd_features']
                    pkd_features['res_features'] = features
                    losses.update(self.losskd(pkd_features, kd_features))
                if 'kd_features' in outputs:
                    outputs.pop('kd_features')
                # bipartite matching-based loss
                losses.update(self.criterion(outputs, targets))

                for k in list(losses.keys()):
                    if k in self.criterion.weight_dict:
                        losses[k] *= self.criterion.weight_dict[k]
                    else:
                        # remove this loss if not specified in `weight_dict`
                        continue
                        # losses.pop(k)
                return losses
            else:
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )

                del outputs

                processed_results = []
                for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                        mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    processed_results.append({})

                    if self.sem_seg_postprocess_before_inference:
                        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            mask_pred_result, image_size, height, width
                        )
                        mask_cls_result = mask_cls_result.to(mask_pred_result)

                    # semantic segmentation inference
                    if self.semantic_on:
                        r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                        if not self.sem_seg_postprocess_before_inference:
                            r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                        processed_results[-1]["sem_seg"] = r

                    # panoptic segmentation inference
                    if self.panoptic_on:
                        panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["panoptic_seg"] = panoptic_r

                    # instance segmentation inference
                    if self.instance_on:
                        instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["instances"] = instance_r

                return processed_results

        def prepare_targets(self, targets, gt_ignores, images):
            h_pad, w_pad = images.tensor.shape[-2:]
            new_targets = []
            for targets_per_image, gt_ignore in zip(targets, gt_ignores):
                # pad gt
                gt_masks = targets_per_image.gt_masks
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
                new_targets.append(
                    {
                        "labels": targets_per_image.gt_classes,
                        "masks": padded_masks,
                        "gt_ignore": gt_ignore,
                    }
                )
            return new_targets

        def semantic_inference(self, mask_cls, mask_pred):

            mask_cls = F.softmax(mask_cls, dim=-1)[..., 1:]  # the first class is the background
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg

        def panoptic_inference(self, mask_cls, mask_pred):
            scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
            mask_pred = mask_pred.sigmoid()

            keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[keep]
            cur_mask_cls = mask_cls[keep]
            cur_mask_cls = cur_mask_cls[:, :-1]

            cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

            h, w = cur_masks.shape[-2:]
            panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
            segments_info = []

            current_segment_id = 0

            if cur_masks.shape[0] == 0:
                # We didn't detect any mask :(
                return panoptic_seg, segments_info
            else:
                # take argmax
                cur_mask_ids = cur_prob_masks.argmax(0)
                stuff_memory_list = {}
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()
                    isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                    mask_area = (cur_mask_ids == k).sum().item()
                    original_area = (cur_masks[k] >= 0.5).sum().item()
                    mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        if mask_area / original_area < self.overlap_threshold:
                            continue

                        # merge stuff regions
                        if not isthing:
                            if int(pred_class) in stuff_memory_list.keys():
                                panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                                continue
                            else:
                                stuff_memory_list[int(pred_class)] = current_segment_id + 1

                        current_segment_id += 1
                        panoptic_seg[mask] = current_segment_id

                        segments_info.append(
                            {
                                "id": current_segment_id,
                                "isthing": bool(isthing),
                                "category_id": int(pred_class),
                            }
                        )

                return panoptic_seg, segments_info

        def instance_inference(self, mask_cls, mask_pred):
            # mask_pred is already processed to have the same shape as original input
            image_size = mask_pred.shape[-2:]

            # [Q, K]
            # scores = F.softmax(mask_cls, dim=-1)[:, :-1]
            scores = F.softmax(mask_cls, dim=-1)[:, 1:]
            num_queries, num_classes = scores.shape
            labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
            # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
            labels_per_image = labels[topk_indices]

            topk_indices = topk_indices // num_classes
            # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
            mask_pred = mask_pred[topk_indices]

            # if this is panoptic segmentation, we only keep the "thing" classes
            if self.panoptic_on:
                keep = torch.zeros_like(scores_per_image).bool()
                for i, lab in enumerate(labels_per_image):
                    keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]
                mask_pred = mask_pred[keep]

            result = Instances(image_size)
            # mask (before sigmoid)
            result.pred_masks = (mask_pred > 0).float()
            result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
            # Uncomment the following to get boxes from masks (this is slow)
            # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

            # calculate average mask prob
            mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
                        result.pred_masks.flatten(1).sum(1) + 1e-6)
            result.scores = scores_per_image * mask_scores_per_image
            result.pred_classes = labels_per_image
            return result

elif version == 'pod':
    # query freeze : True
    # query kd : True
    # prediction kd : True
    # query group attention mask : True


    class SoftDiceLoss(nn.Module):
        def __init__(self, weight=None, size_average=True):
            super(SoftDiceLoss, self).__init__()

        def forward(self, logits, targets):
            bs = targets.size(0)
            smooth = 1.0

            probs = F.sigmoid(logits)
            m1 = probs.view(bs, -1)
            m2 = targets.view(bs, -1)
            intersection = (m1 * m2)

            score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
            score = 1 - score.sum() / bs
            return score


    @META_ARCH_REGISTRY.register()
    class IncrementalMaskFormer(nn.Module):
        """
        Main class for mask classification semantic segmentation architectures.
        """

        @configurable
        def __init__(
                self,
                *,
                cfg: None,
                name: str,
                backbone: Backbone,
                sem_seg_head: nn.Module,
                criterion: nn.Module,
                num_queries: int,
                object_mask_threshold: float,
                overlap_threshold: float,
                metadata,
                size_divisibility: int,
                sem_seg_postprocess_before_inference: bool,
                pixel_mean: Tuple[float],
                pixel_std: Tuple[float],
                # inference
                semantic_on: bool,
                panoptic_on: bool,
                instance_on: bool,
                test_topk_per_image: int,
        ):
            """
            Args:
                backbone: a backbone module, must follow detectron2's backbone interface
                sem_seg_head: a module that predicts semantic segmentation from backbone features
                criterion: a module that defines the loss
                num_queries: int, number of queries
                object_mask_threshold: float, threshold to filter query based on classification score
                    for panoptic segmentation inference
                overlap_threshold: overlap threshold used in general inference for panoptic segmentation
                metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                    segmentation inference
                size_divisibility: Some backbones require the input height and width to be divisible by a
                    specific integer. We can use this to override such requirement.
                sem_seg_postprocess_before_inference: whether to resize the prediction back
                    to original input size before semantic segmentation inference or after.
                    For high-resolution dataset like Mapillary, resizing predictions before
                    inference will cause OOM error.
                pixel_mean, pixel_std: list or tuple with #channels element, representing
                    the per-channel mean and std to be used to normalize the input image
                semantic_on: bool, whether to output semantic segmentation prediction
                instance_on: bool, whether to output instance segmentation prediction
                panoptic_on: bool, whether to output panoptic segmentation prediction
                test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
            """
            super().__init__()
            self.cfg = copy.deepcopy(cfg)
            dataset, _, task, step = name.split('_')
            self.dataset = dataset
            self.task = task
            self.step = int(step)

            sem_seg_head.predictor.task = self.task
            sem_seg_head.predictor.step = self.step


            # freeze the old queries
            if self.step > 0:
                sem_seg_head.predictor.query_feat.weight.requires_grad = False
                for s in range(self.step - 1):
                    sem_seg_head.predictor.incre_query_feat_list[s].weight.requires_grad = False

                # get the teacher model
                if not self.cfg.MODEL.MODEL_OLD:
                    model_old_cfg = copy.deepcopy(cfg)
                    model_old_cfg.defrost()
                    model_old_cfg.MODEL.MODEL_OLD = True

                    model_old = build_model(model_old_cfg)
                    model_old.step = self.step - 1
                    model_old.sem_seg_head.predictor.step = self.step - 1
                    DetectionCheckpointer(model_old, save_dir='./cache/debug').resume_or_load(
                        model_old_cfg.MODEL.WEIGHTS
                    )
                    model_old.eval()
                    self.model_old = {'model_old': model_old}
                else:
                    self.model_old = None


            self.backbone = backbone
            self.sem_seg_head = sem_seg_head
            self.criterion = criterion
            self.num_queries = num_queries
            self.overlap_threshold = overlap_threshold
            self.object_mask_threshold = object_mask_threshold
            self.metadata = metadata
            if size_divisibility < 0:
                # use backbone size_divisibility if not set
                size_divisibility = self.backbone.size_divisibility
            self.size_divisibility = size_divisibility
            self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
            self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

            # additional args
            self.semantic_on = semantic_on
            self.instance_on = instance_on
            self.panoptic_on = panoptic_on
            self.test_topk_per_image = test_topk_per_image

            if not self.semantic_on:
                assert self.sem_seg_postprocess_before_inference

            self.dice_loss = SoftDiceLoss()

            self.params = {
                'collapse_channels': 'local',
                'pod_apply': 'all',
                'pod_factor': 1.,
                'prepro': 'pow',
                'spp_scales': [1, 2, 4],
                'pod_options': {"switch": {"after": {"extra_channels": "sum", "factor": 0.00001, "type": "local"}}},
                'use_pod_schedule': True,
            }





        @classmethod
        def from_config(cls, cfg):
            backbone = build_backbone(cfg)
            sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

            # Loss parameters:
            deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
            no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

            # loss weights
            class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
            dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
            mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

            # building criterion
            matcher = HungarianMatcher(
                cost_class=class_weight,
                cost_mask=mask_weight,
                cost_dice=dice_weight,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            )

            weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

            if deep_supervision:
                dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
                aux_weight_dict = {}
                for i in range(dec_layers - 1):
                    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
                weight_dict.update(aux_weight_dict)

            losses = ["labels", "masks"]

            dataset, seg, task, step = cfg.DATASETS.TRAIN[0].split('_')

            if dataset == 'ade20k':
                if seg == 'incrementalins':
                    from mask2former.data.datasets.register_ade20k_incremental_instance import tasks_ade_ins
                    training_classes = tasks_ade_ins[task][int(step)]
                else:
                    from mask2former.data.datasets.register_ade20k_incremental import tasks_ade
                    training_classes = tasks_ade[task][int(step)]
                cr_num_classes = len(training_classes)
            elif dataset == 'voc':
                from mask2former.data.datasets.register_voc_incremental import tasks_voc
                training_classes = tasks_voc[task][int(step)]
                cr_num_classes = len(training_classes)
            elif dataset == 'coco':
                from mask2former.data.datasets.register_coco_incremental_instance import tasks_coco_ins
                training_classes = tasks_coco_ins[task][int(step)]
                cr_num_classes = len(training_classes)
            else:
                print(f'no dataset {dataset}')
                raise ValueError('No dataset')

            criterion = SetCriterion(
                cr_num_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight,
                losses=losses,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
                oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
                importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            )

            return {
                "cfg": cfg,
                "name": cfg.DATASETS.TRAIN[0],
                "backbone": backbone,
                "sem_seg_head": sem_seg_head,
                "criterion": criterion,
                "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
                "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
                "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
                "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
                "sem_seg_postprocess_before_inference": (
                        cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                        or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                        or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
                ),
                "pixel_mean": cfg.MODEL.PIXEL_MEAN,
                "pixel_std": cfg.MODEL.PIXEL_STD,
                # inference
                "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
                "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
                "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
                "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            }

        @property
        def device(self):
            return self.pixel_mean.device


        def losskd(self, new_features, old_features):
            losskd = dict()

            # Prediction KD
            predictions_classs = old_features['predictions_class']
            ppredictions_classs = new_features['predictions_class']
            predictions_masks = old_features['predictions_mask']
            ppredictions_masks = new_features['predictions_mask']
            nl = [9]
            nk = predictions_classs[0].keys()
            for l in nl:
                total_nq = sum([ppredictions_classs[l][k].shape[1] for k in nk])
                losskd[f'loss_kd_class{l}'] = 0
                losskd[f'loss_kd_mask{l}'] = 0
                for k in nk:
                    pred_class = ppredictions_classs[l][k]
                    gt_class = predictions_classs[l][k]

                    b, nq, c = pred_class.shape

                    pred = pred_class.reshape(-1, c).softmax(-1)
                    gt = gt_class.reshape(-1, c).softmax(-1)
                    # 1000 for 11 task, 300 for 2 or 3 task
                    step_loss_kd_class = (F.kl_div(pred.log(), gt, reduction='sum') / (b * nq) * 300) * nq / total_nq
                    losskd[f'loss_kd_class{l}'] += step_loss_kd_class

                    pred_mask = ppredictions_masks[l][k]
                    gt_mask = predictions_masks[l][k]

                    b, nq, h, w = pred_mask.shape
                    pred = pred_mask.reshape(-1, h, w)
                    gt = (gt_mask.reshape(-1, h, w) > 0).to(torch.int64)
                    step_loss_kd_mask = (self.dice_loss(pred, gt) * 300) * nq / total_nq
                    losskd[f'loss_kd_mask{l}'] += step_loss_kd_mask




            pod_old_features = []
            for k, v in old_features['res_features'].items():
                pod_old_features.append(v)
            pod_old_features.append(old_features['mask_features'])
            pod_new_features = []
            for k, v in new_features['res_features'].items():
                pod_new_features.append(v)
            pod_new_features.append(new_features['mask_features'])


            losskd[f'loss_kd_lpod'] = features_distillation(pod_old_features, pod_new_features, **self.params)


            return losskd

        def detach_features(self, features):
            if isinstance(features, torch.Tensor):
                return features.detach()
            elif isinstance(features, dict):
                new_dict = dict()
                for k, v in features.items():
                    new_dict[k] = self.detach_features(v)
                return new_dict
            elif isinstance(features, list):
                return [self.detach_features(el) for el in features]
            else:
                raise ValueError("unknow type")

        def forward(self, batched_inputs, kd=False):
            if kd:
                return self.forward_old(batched_inputs)
            else:
                return self.forward_new(batched_inputs)

        def forward_old(self, batched_inputs):
            # forward path of the model
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            features = self.backbone(images.tensor)

            outputs = self.sem_seg_head(features)

            # semantic predictions
            sem_logits = self.semantic_inference(outputs['pred_logits'], outputs['pred_masks'])
            outputs['kd_features']['predictions_sem'] = sem_logits

            # return the kd features
            kd_features = outputs['kd_features']
            kd_features['res_features'] = features
            return self.detach_features(kd_features)

        def forward_new(self, batched_inputs):
            # forward path of the model
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            features = self.backbone(images.tensor)

            outputs = self.sem_seg_head(features)

            # semantic predictions
            # sem_logits = self.semantic_inference(outputs['pred_logits'], outputs['pred_masks'])
            # outputs['kd_features']['predictions_sem'] = sem_logits

            if self.training:
                # mask classification target
                if "instances" in batched_inputs[0]:
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                    gt_ignores = [x["ignore"].to(self.device) for x in batched_inputs]
                    targets = self.prepare_targets(gt_instances, gt_ignores, images)
                else:
                    targets = None

                losses = dict()

                # cal the kd loss
                if self.training and self.model_old is not None:
                    old_features = self.model_old['model_old'](batched_inputs, kd=True)
                    new_features = outputs['kd_features']
                    new_features['res_features'] = features
                    losses.update(self.losskd(new_features, old_features))

                if 'kd_features' in outputs:
                    outputs.pop('kd_features')

                # bipartite matching-based loss
                losses.update(self.criterion(outputs, targets))

                for k in list(losses.keys()):
                    if k in self.criterion.weight_dict:
                        losses[k] *= self.criterion.weight_dict[k]
                    else:
                        # remove this loss if not specified in `weight_dict`
                        continue
                        # losses.pop(k)
                return losses
            else:
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )

                del outputs

                processed_results = []
                for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                        mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    processed_results.append({})

                    if self.sem_seg_postprocess_before_inference:
                        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            mask_pred_result, image_size, height, width
                        )
                        mask_cls_result = mask_cls_result.to(mask_pred_result)

                    # semantic segmentation inference
                    if self.semantic_on:
                        r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                        if not self.sem_seg_postprocess_before_inference:
                            r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                        processed_results[-1]["sem_seg"] = r

                    # panoptic segmentation inference
                    if self.panoptic_on:
                        panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["panoptic_seg"] = panoptic_r

                    # instance segmentation inference
                    if self.instance_on:
                        instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["instances"] = instance_r

                return processed_results

        def prepare_targets(self, targets, gt_ignores, images):
            h_pad, w_pad = images.tensor.shape[-2:]
            new_targets = []
            for targets_per_image, gt_ignore in zip(targets, gt_ignores):
                # pad gt
                gt_masks = targets_per_image.gt_masks
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype,
                                           device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
                new_targets.append(
                    {
                        "labels": targets_per_image.gt_classes,
                        "masks": padded_masks,
                        "gt_ignore": gt_ignore,
                    }
                )
            return new_targets

        # def semantic_inference(self, mask_cls, mask_pred):
        #     splits = [0,100,110,120,130]
        #     l = len(splits)-1
        #     semseg_list = []
        #     bg_semseg_all = torch.ones((l,*mask_pred.shape[-2:])).to(mask_pred)
        #     for i in range(l):
        #         st, ed = splits[i], splits[i+1]
        #         mask_cls_i = F.softmax(mask_cls[st:ed], dim=-1)[:,1:] # the first class is the background
        #         mask_pred_i = mask_pred[st:ed].sigmoid()
        #         semseg = torch.einsum("qc,qhw->chw", mask_cls_i, mask_pred_i)
        #         fg_semseg = (mask_cls_i.max(1).values[:,None,None]*mask_pred_i).max(0).values
        #         bg_semseg = 1-fg_semseg
        #
        #         # if i==0:
        #         bg_semseg_all = bg_semseg_all*(bg_semseg+1e-6)
        #         bg_semseg_all[i] = bg_semseg_all[i]*(fg_semseg+1e-6)/(bg_semseg+1e-6)
        #
        #         semseg_list.append(semseg)
        #
        #     semseg = torch.stack(semseg_list,0)
        #     bg_semseg_all = bg_semseg_all.softmax(0)
        #
        #     semseg = torch.einsum("schw,shw->chw",semseg, bg_semseg_all)
        #
        #     return semseg

        def semantic_inference(self, mask_cls, mask_pred):

            mask_cls = F.softmax(mask_cls, dim=-1)[..., 1:]  # the first class is the background
            mask_pred = mask_pred.sigmoid()
            if len(mask_cls.shape)==2:
                semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            else:
                semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
            return semseg

        def panoptic_inference(self, mask_cls, mask_pred):
            scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
            mask_pred = mask_pred.sigmoid()

            keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[keep]
            cur_mask_cls = mask_cls[keep]
            cur_mask_cls = cur_mask_cls[:, :-1]

            cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

            h, w = cur_masks.shape[-2:]
            panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
            segments_info = []

            current_segment_id = 0

            if cur_masks.shape[0] == 0:
                # We didn't detect any mask :(
                return panoptic_seg, segments_info
            else:
                # take argmax
                cur_mask_ids = cur_prob_masks.argmax(0)
                stuff_memory_list = {}
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()
                    isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                    mask_area = (cur_mask_ids == k).sum().item()
                    original_area = (cur_masks[k] >= 0.5).sum().item()
                    mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        if mask_area / original_area < self.overlap_threshold:
                            continue

                        # merge stuff regions
                        if not isthing:
                            if int(pred_class) in stuff_memory_list.keys():
                                panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                                continue
                            else:
                                stuff_memory_list[int(pred_class)] = current_segment_id + 1

                        current_segment_id += 1
                        panoptic_seg[mask] = current_segment_id

                        segments_info.append(
                            {
                                "id": current_segment_id,
                                "isthing": bool(isthing),
                                "category_id": int(pred_class),
                            }
                        )

                return panoptic_seg, segments_info

        def instance_inference(self, mask_cls, mask_pred):
            # mask_pred is already processed to have the same shape as original input
            image_size = mask_pred.shape[-2:]

            # [Q, K]
            # scores = F.softmax(mask_cls, dim=-1)[:, :-1]
            scores = F.softmax(mask_cls, dim=-1)[:, 1:]
            num_queries, num_classes = scores.shape
            labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
            # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
            labels_per_image = labels[topk_indices]

            topk_indices = topk_indices // num_classes
            # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
            mask_pred = mask_pred[topk_indices]

            # if this is panoptic segmentation, we only keep the "thing" classes
            if self.panoptic_on:
                keep = torch.zeros_like(scores_per_image).bool()
                for i, lab in enumerate(labels_per_image):
                    keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]
                mask_pred = mask_pred[keep]

            result = Instances(image_size)
            # mask (before sigmoid)
            result.pred_masks = (mask_pred > 0).float()
            result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
            # Uncomment the following to get boxes from masks (this is slow)
            # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

            # calculate average mask prob
            mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
                    result.pred_masks.flatten(1).sum(1) + 1e-6)
            result.scores = scores_per_image * mask_scores_per_image
            result.pred_classes = labels_per_image
            return result

elif version == 'freeze':
    # query freeze : True
    # query kd : True
    # prediction kd : True
    # query group attention mask : True


    @META_ARCH_REGISTRY.register()
    class IncrementalMaskFormer(nn.Module):
        """
        Main class for mask classification semantic segmentation architectures.
        """

        @configurable
        def __init__(
                self,
                *,
                cfg: None,
                name: str,
                backbone: Backbone,
                sem_seg_head: nn.Module,
                criterion: nn.Module,
                num_queries: int,
                object_mask_threshold: float,
                overlap_threshold: float,
                metadata,
                size_divisibility: int,
                sem_seg_postprocess_before_inference: bool,
                pixel_mean: Tuple[float],
                pixel_std: Tuple[float],
                # inference
                semantic_on: bool,
                panoptic_on: bool,
                instance_on: bool,
                test_topk_per_image: int,
        ):
            """
            Args:
                backbone: a backbone module, must follow detectron2's backbone interface
                sem_seg_head: a module that predicts semantic segmentation from backbone features
                criterion: a module that defines the loss
                num_queries: int, number of queries
                object_mask_threshold: float, threshold to filter query based on classification score
                    for panoptic segmentation inference
                overlap_threshold: overlap threshold used in general inference for panoptic segmentation
                metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                    segmentation inference
                size_divisibility: Some backbones require the input height and width to be divisible by a
                    specific integer. We can use this to override such requirement.
                sem_seg_postprocess_before_inference: whether to resize the prediction back
                    to original input size before semantic segmentation inference or after.
                    For high-resolution dataset like Mapillary, resizing predictions before
                    inference will cause OOM error.
                pixel_mean, pixel_std: list or tuple with #channels element, representing
                    the per-channel mean and std to be used to normalize the input image
                semantic_on: bool, whether to output semantic segmentation prediction
                instance_on: bool, whether to output instance segmentation prediction
                panoptic_on: bool, whether to output panoptic segmentation prediction
                test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
            """
            super().__init__()
            self.cfg = copy.deepcopy(cfg)
            dataset, _, task, step = name.split('_')
            self.dataset = dataset
            self.task = task
            self.step = int(step)

            sem_seg_head.predictor.task = self.task
            sem_seg_head.predictor.step = self.step



            self.backbone = backbone
            self.sem_seg_head = sem_seg_head
            self.criterion = criterion
            self.num_queries = num_queries
            self.overlap_threshold = overlap_threshold
            self.object_mask_threshold = object_mask_threshold
            self.metadata = metadata
            if size_divisibility < 0:
                # use backbone size_divisibility if not set
                size_divisibility = self.backbone.size_divisibility
            self.size_divisibility = size_divisibility
            self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
            self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

            # additional args
            self.semantic_on = semantic_on
            self.instance_on = instance_on
            self.panoptic_on = panoptic_on
            self.test_topk_per_image = test_topk_per_image

            if not self.semantic_on:
                assert self.sem_seg_postprocess_before_inference


            # incremental setting
            trainable_params = []
            if self.step!=0:
                for name,param in self.named_parameters():
                    if 'incre' not in name:
                        param.requires_grad = False
                    else:
                        trainable_params.append(name)
                print('#'*50)
                print('Trainable params :')
                print(trainable_params)
                print('#' * 50)





        @classmethod
        def from_config(cls, cfg):
            backbone = build_backbone(cfg)
            sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

            # Loss parameters:
            deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
            no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

            # loss weights
            class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
            dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
            mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

            # building criterion
            matcher = HungarianMatcher(
                cost_class=class_weight,
                cost_mask=mask_weight,
                cost_dice=dice_weight,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            )

            weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

            if deep_supervision:
                dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
                aux_weight_dict = {}
                for i in range(dec_layers - 1):
                    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
                weight_dict.update(aux_weight_dict)

            losses = ["labels", "masks"]

            dataset, seg, task, step = cfg.DATASETS.TRAIN[0].split('_')

            if dataset == 'ade20k':
                if seg == 'incrementalins':
                    from mask2former.data.datasets.register_ade20k_incremental_instance import tasks_ade_ins
                    training_classes = tasks_ade_ins[task][int(step)]
                else:
                    from mask2former.data.datasets.register_ade20k_incremental import tasks_ade
                    training_classes = tasks_ade[task][int(step)]
                cr_num_classes = len(training_classes)
            elif dataset == 'voc':
                from mask2former.data.datasets.register_voc_incremental import tasks_voc
                training_classes = tasks_voc[task][int(step)]
                cr_num_classes = len(training_classes)
            elif dataset == 'coco':
                from mask2former.data.datasets.register_coco_incremental_instance import tasks_coco_ins
                training_classes = tasks_coco_ins[task][int(step)]
                cr_num_classes = len(training_classes)
            else:
                print(f'no dataset {dataset}')
                raise ValueError('No dataset')

            criterion = SetCriterion(
                cr_num_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight,
                losses=losses,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
                oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
                importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            )

            return {
                "cfg": cfg,
                "name": cfg.DATASETS.TRAIN[0],
                "backbone": backbone,
                "sem_seg_head": sem_seg_head,
                "criterion": criterion,
                "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
                "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
                "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
                "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
                "sem_seg_postprocess_before_inference": (
                        cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                        or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                        or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
                ),
                "pixel_mean": cfg.MODEL.PIXEL_MEAN,
                "pixel_std": cfg.MODEL.PIXEL_STD,
                # inference
                "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
                "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
                "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
                "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            }

        @property
        def device(self):
            return self.pixel_mean.device

        def detach_features(self, features):
            if isinstance(features, torch.Tensor):
                return features.detach()
            elif isinstance(features, dict):
                new_dict = dict()
                for k, v in features.items():
                    new_dict[k] = self.detach_features(v)
                return new_dict
            elif isinstance(features, list):
                return [self.detach_features(el) for el in features]
            else:
                raise ValueError("unknow type")

        def forward(self, batched_inputs):
            # forward path of the model
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            features = self.backbone(images.tensor)

            outputs = self.sem_seg_head(features)

            # semantic predictions
            # sem_logits = self.semantic_inference(outputs['pred_logits'], outputs['pred_masks'])
            # outputs['kd_features']['predictions_sem'] = sem_logits

            if self.training:
                # mask classification target
                if "instances" in batched_inputs[0]:
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                    gt_ignores = [x["ignore"].to(self.device) for x in batched_inputs]
                    targets = self.prepare_targets(gt_instances, gt_ignores, images)
                else:
                    targets = None

                losses = dict()

                if 'kd_features' in outputs:
                    outputs.pop('kd_features')

                # bipartite matching-based loss
                losses.update(self.criterion(outputs, targets))

                for k in list(losses.keys()):
                    if k in self.criterion.weight_dict:
                        losses[k] *= self.criterion.weight_dict[k]
                    else:
                        # remove this loss if not specified in `weight_dict`
                        continue
                        # losses.pop(k)
                return losses
            else:
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )

                del outputs

                processed_results = []
                for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                        mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    processed_results.append({})

                    if self.sem_seg_postprocess_before_inference:
                        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            mask_pred_result, image_size, height, width
                        )
                        mask_cls_result = mask_cls_result.to(mask_pred_result)

                    # semantic segmentation inference
                    if self.semantic_on:
                        r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                        if not self.sem_seg_postprocess_before_inference:
                            r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                        processed_results[-1]["sem_seg"] = r

                    # panoptic segmentation inference
                    if self.panoptic_on:
                        panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["panoptic_seg"] = panoptic_r

                    # instance segmentation inference
                    if self.instance_on:
                        instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["instances"] = instance_r

                return processed_results

        def prepare_targets(self, targets, gt_ignores, images):
            h_pad, w_pad = images.tensor.shape[-2:]
            new_targets = []
            for targets_per_image, gt_ignore in zip(targets, gt_ignores):
                # pad gt
                gt_masks = targets_per_image.gt_masks
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype,
                                           device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
                new_targets.append(
                    {
                        "labels": targets_per_image.gt_classes,
                        "masks": padded_masks,
                        "gt_ignore": gt_ignore,
                    }
                )
            return new_targets

        def semantic_inference(self, mask_cls, mask_pred):

            # mask_cls = mask_cls[100:120]
            # mask_pred = mask_pred[100:120]
            # mask_cls = mask_cls[100:120]
            # mask_pred = mask_pred[100:120]
            mask_cls = F.softmax(mask_cls, dim=-1)[..., 1:]  # the first class is the background
            mask_pred = mask_pred.sigmoid()
            if len(mask_cls.shape)==2:
                semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            else:
                semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
            # n_semseg = -torch.ones_like(semseg)*1e6
            # n_semseg[100:110] = semseg[100:110]
            return semseg

        # def semantic_inference(self, mask_cls, mask_pred):
        #     splits = [0,100,110,120,130]
        #     l = len(splits)-1
        #     semseg_list = []
        #     bg_semseg_all = torch.ones((l,*mask_pred.shape[-2:])).to(mask_pred)
        #     for i in range(l):
        #         st, ed = splits[i], splits[i+1]
        #         mask_cls_i = F.softmax(mask_cls[st:ed], dim=-1)[:,1:] # the first class is the background
        #         mask_pred_i = mask_pred[st:ed].sigmoid()
        #         semseg = torch.einsum("qc,qhw->chw", mask_cls_i, mask_pred_i)
        #         fg_semseg = (mask_cls_i.max(1).values[:,None,None]*mask_pred_i).max(0).values
        #         bg_semseg = 1-fg_semseg
        #
        #         if i==0:
        #             bg_semseg_all = bg_semseg_all*(bg_semseg+1e-6)
        #             bg_semseg_all[i] = bg_semseg_all[i]*fg_semseg/(bg_semseg+1e-6)
        #
        #         semseg_list.append(semseg)
        #
        #     semseg = torch.stack(semseg_list,0)
        #     bg_semseg_all = bg_semseg_all.softmax(0)
        #
        #     semseg = torch.einsum("schw,shw->chw",semseg, bg_semseg_all)
        #
        #     return semseg

        def panoptic_inference(self, mask_cls, mask_pred):
            scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
            mask_pred = mask_pred.sigmoid()

            keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[keep]
            cur_mask_cls = mask_cls[keep]
            cur_mask_cls = cur_mask_cls[:, :-1]

            cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

            h, w = cur_masks.shape[-2:]
            panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
            segments_info = []

            current_segment_id = 0

            if cur_masks.shape[0] == 0:
                # We didn't detect any mask :(
                return panoptic_seg, segments_info
            else:
                # take argmax
                cur_mask_ids = cur_prob_masks.argmax(0)
                stuff_memory_list = {}
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()
                    isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                    mask_area = (cur_mask_ids == k).sum().item()
                    original_area = (cur_masks[k] >= 0.5).sum().item()
                    mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        if mask_area / original_area < self.overlap_threshold:
                            continue

                        # merge stuff regions
                        if not isthing:
                            if int(pred_class) in stuff_memory_list.keys():
                                panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                                continue
                            else:
                                stuff_memory_list[int(pred_class)] = current_segment_id + 1

                        current_segment_id += 1
                        panoptic_seg[mask] = current_segment_id

                        segments_info.append(
                            {
                                "id": current_segment_id,
                                "isthing": bool(isthing),
                                "category_id": int(pred_class),
                            }
                        )

                return panoptic_seg, segments_info

        def instance_inference(self, mask_cls, mask_pred):
            # mask_pred is already processed to have the same shape as original input
            image_size = mask_pred.shape[-2:]

            # [Q, K]
            # scores = F.softmax(mask_cls, dim=-1)[:, :-1]
            scores = F.softmax(mask_cls, dim=-1)[:, 1:]
            num_queries, num_classes = scores.shape
            labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
            # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
            labels_per_image = labels[topk_indices]

            topk_indices = topk_indices // num_classes
            # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
            mask_pred = mask_pred[topk_indices]

            # if this is panoptic segmentation, we only keep the "thing" classes
            if self.panoptic_on:
                keep = torch.zeros_like(scores_per_image).bool()
                for i, lab in enumerate(labels_per_image):
                    keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]
                mask_pred = mask_pred[keep]

            result = Instances(image_size)
            # mask (before sigmoid)
            result.pred_masks = (mask_pred > 0).float()
            result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
            # Uncomment the following to get boxes from masks (this is slow)
            # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

            # calculate average mask prob
            mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
                    result.pred_masks.flatten(1).sum(1) + 1e-6)
            result.scores = scores_per_image * mask_scores_per_image
            result.pred_classes = labels_per_image
            return result

elif version == 'freeze_old':
    # query freeze : True
    # query kd : True
    # prediction kd : True
    # query group attention mask : True


    @META_ARCH_REGISTRY.register()
    class IncrementalMaskFormer(nn.Module):
        """
        Main class for mask classification semantic segmentation architectures.
        """

        @configurable
        def __init__(
                self,
                *,
                cfg: None,
                name: str,
                backbone: Backbone,
                sem_seg_head: nn.Module,
                criterion: nn.Module,
                num_queries: int,
                object_mask_threshold: float,
                overlap_threshold: float,
                metadata,
                size_divisibility: int,
                sem_seg_postprocess_before_inference: bool,
                pixel_mean: Tuple[float],
                pixel_std: Tuple[float],
                # inference
                semantic_on: bool,
                panoptic_on: bool,
                instance_on: bool,
                test_topk_per_image: int,
        ):
            """
            Args:
                backbone: a backbone module, must follow detectron2's backbone interface
                sem_seg_head: a module that predicts semantic segmentation from backbone features
                criterion: a module that defines the loss
                num_queries: int, number of queries
                object_mask_threshold: float, threshold to filter query based on classification score
                    for panoptic segmentation inference
                overlap_threshold: overlap threshold used in general inference for panoptic segmentation
                metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                    segmentation inference
                size_divisibility: Some backbones require the input height and width to be divisible by a
                    specific integer. We can use this to override such requirement.
                sem_seg_postprocess_before_inference: whether to resize the prediction back
                    to original input size before semantic segmentation inference or after.
                    For high-resolution dataset like Mapillary, resizing predictions before
                    inference will cause OOM error.
                pixel_mean, pixel_std: list or tuple with #channels element, representing
                    the per-channel mean and std to be used to normalize the input image
                semantic_on: bool, whether to output semantic segmentation prediction
                instance_on: bool, whether to output instance segmentation prediction
                panoptic_on: bool, whether to output panoptic segmentation prediction
                test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
            """
            super().__init__()
            self.cfg = copy.deepcopy(cfg)
            dataset, _, task, step = name.split('_')
            self.dataset = dataset
            self.task = task
            self.step = int(step)

            sem_seg_head.predictor.task = self.task
            sem_seg_head.predictor.step = self.step



            self.backbone = backbone
            self.sem_seg_head = sem_seg_head
            self.criterion = criterion
            self.num_queries = num_queries
            self.overlap_threshold = overlap_threshold
            self.object_mask_threshold = object_mask_threshold
            self.metadata = metadata
            if size_divisibility < 0:
                # use backbone size_divisibility if not set
                size_divisibility = self.backbone.size_divisibility
            self.size_divisibility = size_divisibility
            self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
            self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

            # additional args
            self.semantic_on = semantic_on
            self.instance_on = instance_on
            self.panoptic_on = panoptic_on
            self.test_topk_per_image = test_topk_per_image

            if not self.semantic_on:
                assert self.sem_seg_postprocess_before_inference


            # incremental setting
            trainable_params = []
            if self.step!=0:
                for name,param in self.named_parameters():
                    if 'incre' not in name:
                        param.requires_grad = False
                    else:
                        trainable_params.append(name)
                print('#'*50)
                print('Trainable params :')
                print(trainable_params)
                print('#' * 50)





        @classmethod
        def from_config(cls, cfg):
            backbone = build_backbone(cfg)
            sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

            # Loss parameters:
            deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
            no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

            # loss weights
            class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
            dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
            mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

            # building criterion
            matcher = HungarianMatcher(
                cost_class=class_weight,
                cost_mask=mask_weight,
                cost_dice=dice_weight,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            )

            weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

            if deep_supervision:
                dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
                aux_weight_dict = {}
                for i in range(dec_layers - 1):
                    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
                weight_dict.update(aux_weight_dict)

            losses = ["labels", "masks"]

            dataset, seg, task, step = cfg.DATASETS.TRAIN[0].split('_')

            if dataset == 'ade20k':
                if seg == 'incrementalins':
                    from mask2former.data.datasets.register_ade20k_incremental_instance import tasks_ade_ins
                    training_classes = tasks_ade_ins[task][int(step)]
                else:
                    from mask2former.data.datasets.register_ade20k_incremental import tasks_ade
                    training_classes = tasks_ade[task][int(step)]
                cr_num_classes = len(training_classes)
            elif dataset == 'voc':
                from mask2former.data.datasets.register_voc_incremental import tasks_voc
                training_classes = tasks_voc[task][int(step)]
                cr_num_classes = len(training_classes)
            elif dataset == 'coco':
                from mask2former.data.datasets.register_coco_incremental_instance import tasks_coco_ins
                training_classes = tasks_coco_ins[task][int(step)]
                cr_num_classes = len(training_classes)
            else:
                print(f'no dataset {dataset}')
                raise ValueError('No dataset')

            criterion = SetCriterion(
                cr_num_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight,
                losses=losses,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
                oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
                importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            )

            return {
                "cfg": cfg,
                "name": cfg.DATASETS.TRAIN[0],
                "backbone": backbone,
                "sem_seg_head": sem_seg_head,
                "criterion": criterion,
                "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
                "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
                "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
                "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
                "sem_seg_postprocess_before_inference": (
                        cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                        or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                        or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
                ),
                "pixel_mean": cfg.MODEL.PIXEL_MEAN,
                "pixel_std": cfg.MODEL.PIXEL_STD,
                # inference
                "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
                "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
                "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
                "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            }

        @property
        def device(self):
            return self.pixel_mean.device

        def detach_features(self, features):
            if isinstance(features, torch.Tensor):
                return features.detach()
            elif isinstance(features, dict):
                new_dict = dict()
                for k, v in features.items():
                    new_dict[k] = self.detach_features(v)
                return new_dict
            elif isinstance(features, list):
                return [self.detach_features(el) for el in features]
            else:
                raise ValueError("unknow type")

        def forward(self, batched_inputs):
            # forward path of the model
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            features = self.backbone(images.tensor)

            outputs = self.sem_seg_head(features)

            # semantic predictions
            sem_logits = self.semantic_inference(outputs['pred_logits'], outputs['pred_masks'])
            outputs['kd_features']['predictions_sem'] = sem_logits

            if self.training:
                # mask classification target
                if "instances" in batched_inputs[0]:
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                    gt_ignores = [x["ignore"].to(self.device) for x in batched_inputs]
                    targets = self.prepare_targets(gt_instances, gt_ignores, images)
                else:
                    targets = None

                losses = dict()

                if 'kd_features' in outputs:
                    outputs.pop('kd_features')

                # bipartite matching-based loss
                losses.update(self.criterion(outputs, targets))

                for k in list(losses.keys()):
                    if k in self.criterion.weight_dict:
                        losses[k] *= self.criterion.weight_dict[k]
                    else:
                        # remove this loss if not specified in `weight_dict`
                        continue
                        # losses.pop(k)
                return losses
            else:
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )

                del outputs

                processed_results = []
                for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                        mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    processed_results.append({})

                    if self.sem_seg_postprocess_before_inference:
                        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            mask_pred_result, image_size, height, width
                        )
                        mask_cls_result = mask_cls_result.to(mask_pred_result)

                    # semantic segmentation inference
                    if self.semantic_on:
                        r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                        if not self.sem_seg_postprocess_before_inference:
                            r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                        processed_results[-1]["sem_seg"] = r

                    # panoptic segmentation inference
                    if self.panoptic_on:
                        panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["panoptic_seg"] = panoptic_r

                    # instance segmentation inference
                    if self.instance_on:
                        instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["instances"] = instance_r

                return processed_results

        def prepare_targets(self, targets, gt_ignores, images):
            h_pad, w_pad = images.tensor.shape[-2:]
            new_targets = []
            for targets_per_image, gt_ignore in zip(targets, gt_ignores):
                # pad gt
                gt_masks = targets_per_image.gt_masks
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype,
                                           device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
                new_targets.append(
                    {
                        "labels": targets_per_image.gt_classes,
                        "masks": padded_masks,
                        "gt_ignore": gt_ignore,
                    }
                )
            return new_targets

        def semantic_inference(self, mask_cls, mask_pred):

            mask_cls = F.softmax(mask_cls, dim=-1)[..., 1:]  # the first class is the background
            mask_pred = mask_pred.sigmoid()
            if len(mask_cls.shape)==2:
                semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            else:
                semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
            return semseg

        def panoptic_inference(self, mask_cls, mask_pred):
            scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
            mask_pred = mask_pred.sigmoid()

            keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[keep]
            cur_mask_cls = mask_cls[keep]
            cur_mask_cls = cur_mask_cls[:, :-1]

            cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

            h, w = cur_masks.shape[-2:]
            panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
            segments_info = []

            current_segment_id = 0

            if cur_masks.shape[0] == 0:
                # We didn't detect any mask :(
                return panoptic_seg, segments_info
            else:
                # take argmax
                cur_mask_ids = cur_prob_masks.argmax(0)
                stuff_memory_list = {}
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()
                    isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                    mask_area = (cur_mask_ids == k).sum().item()
                    original_area = (cur_masks[k] >= 0.5).sum().item()
                    mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        if mask_area / original_area < self.overlap_threshold:
                            continue

                        # merge stuff regions
                        if not isthing:
                            if int(pred_class) in stuff_memory_list.keys():
                                panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                                continue
                            else:
                                stuff_memory_list[int(pred_class)] = current_segment_id + 1

                        current_segment_id += 1
                        panoptic_seg[mask] = current_segment_id

                        segments_info.append(
                            {
                                "id": current_segment_id,
                                "isthing": bool(isthing),
                                "category_id": int(pred_class),
                            }
                        )

                return panoptic_seg, segments_info

        def instance_inference(self, mask_cls, mask_pred):
            # mask_pred is already processed to have the same shape as original input
            image_size = mask_pred.shape[-2:]

            # [Q, K]
            # scores = F.softmax(mask_cls, dim=-1)[:, :-1]
            scores = F.softmax(mask_cls, dim=-1)[:, 1:]
            num_queries, num_classes = scores.shape
            labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
            # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
            labels_per_image = labels[topk_indices]

            topk_indices = topk_indices // num_classes
            # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
            mask_pred = mask_pred[topk_indices]

            # if this is panoptic segmentation, we only keep the "thing" classes
            if self.panoptic_on:
                keep = torch.zeros_like(scores_per_image).bool()
                for i, lab in enumerate(labels_per_image):
                    keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]
                mask_pred = mask_pred[keep]

            result = Instances(image_size)
            # mask (before sigmoid)
            result.pred_masks = (mask_pred > 0).float()
            result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
            # Uncomment the following to get boxes from masks (this is slow)
            # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

            # calculate average mask prob
            mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
                    result.pred_masks.flatten(1).sum(1) + 1e-6)
            result.scores = scores_per_image * mask_scores_per_image
            result.pred_classes = labels_per_image
            return result

elif version == 'freeze_pod':
    # query freeze : True
    # query kd : True
    # prediction kd : True
    # query group attention mask : True


    class SoftDiceLoss(nn.Module):
        def __init__(self, weight=None, size_average=True):
            super(SoftDiceLoss, self).__init__()

        def forward(self, logits, targets):
            bs = targets.size(0)
            smooth = 1.0

            probs = F.sigmoid(logits)
            m1 = probs.view(bs, -1)
            m2 = targets.view(bs, -1)
            intersection = (m1 * m2)

            score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
            score = 1 - score.sum() / bs
            return score


    @META_ARCH_REGISTRY.register()
    class IncrementalMaskFormer(nn.Module):
        """
        Main class for mask classification semantic segmentation architectures.
        """

        @configurable
        def __init__(
                self,
                *,
                cfg: None,
                name: str,
                backbone: Backbone,
                sem_seg_head: nn.Module,
                criterion: nn.Module,
                num_queries: int,
                object_mask_threshold: float,
                overlap_threshold: float,
                metadata,
                size_divisibility: int,
                sem_seg_postprocess_before_inference: bool,
                pixel_mean: Tuple[float],
                pixel_std: Tuple[float],
                # inference
                semantic_on: bool,
                panoptic_on: bool,
                instance_on: bool,
                test_topk_per_image: int,
        ):
            """
            Args:
                backbone: a backbone module, must follow detectron2's backbone interface
                sem_seg_head: a module that predicts semantic segmentation from backbone features
                criterion: a module that defines the loss
                num_queries: int, number of queries
                object_mask_threshold: float, threshold to filter query based on classification score
                    for panoptic segmentation inference
                overlap_threshold: overlap threshold used in general inference for panoptic segmentation
                metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                    segmentation inference
                size_divisibility: Some backbones require the input height and width to be divisible by a
                    specific integer. We can use this to override such requirement.
                sem_seg_postprocess_before_inference: whether to resize the prediction back
                    to original input size before semantic segmentation inference or after.
                    For high-resolution dataset like Mapillary, resizing predictions before
                    inference will cause OOM error.
                pixel_mean, pixel_std: list or tuple with #channels element, representing
                    the per-channel mean and std to be used to normalize the input image
                semantic_on: bool, whether to output semantic segmentation prediction
                instance_on: bool, whether to output instance segmentation prediction
                panoptic_on: bool, whether to output panoptic segmentation prediction
                test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
            """
            super().__init__()
            self.cfg = copy.deepcopy(cfg)
            dataset, _, task, step = name.split('_')
            self.dataset = dataset
            self.task = task
            self.step = int(step)

            sem_seg_head.predictor.task = self.task
            sem_seg_head.predictor.step = self.step


            # freeze the old queries
            if self.step > 0:
                sem_seg_head.predictor.query_feat.weight.requires_grad = False
                for s in range(self.step - 1):
                    sem_seg_head.predictor.incre_query_feat_list[s].weight.requires_grad = False

                # get the teacher model
                if not self.cfg.MODEL.MODEL_OLD:
                    model_old_cfg = copy.deepcopy(cfg)
                    model_old_cfg.defrost()
                    model_old_cfg.MODEL.MODEL_OLD = True

                    model_old = build_model(model_old_cfg)
                    model_old.step = self.step - 1
                    model_old.sem_seg_head.predictor.step = self.step - 1
                    DetectionCheckpointer(model_old, save_dir='./cache/debug').resume_or_load(
                        model_old_cfg.MODEL.WEIGHTS
                    )
                    model_old.eval()
                    self.model_old = {'model_old': model_old}
                else:
                    self.model_old = None


            self.backbone = backbone
            self.sem_seg_head = sem_seg_head
            self.criterion = criterion
            self.num_queries = num_queries
            self.overlap_threshold = overlap_threshold
            self.object_mask_threshold = object_mask_threshold
            self.metadata = metadata
            if size_divisibility < 0:
                # use backbone size_divisibility if not set
                size_divisibility = self.backbone.size_divisibility
            self.size_divisibility = size_divisibility
            self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
            self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

            # additional args
            self.semantic_on = semantic_on
            self.instance_on = instance_on
            self.panoptic_on = panoptic_on
            self.test_topk_per_image = test_topk_per_image

            if not self.semantic_on:
                assert self.sem_seg_postprocess_before_inference

            self.dice_loss = SoftDiceLoss()


            # incremental setting
            trainable_params = []
            if self.step!=0:
                for name,param in self.named_parameters():
                    if 'backbone' in name:
                        param.requires_grad = False
                    else:
                        trainable_params.append(name)
                    # if 'incre' not in name:
                    #     param.requires_grad = False
                    # else:
                    #     trainable_params.append(name)
                print('#'*50)
                print('Trainable params :')
                print(trainable_params)
                print('#' * 50)


            self.params = {
                'collapse_channels': 'local',
                'pod_apply': 'all',
                'pod_factor': 1.,
                'prepro': 'pow',
                'spp_scales': [1, 2, 4],
                'pod_options': {"switch": {"after": {"extra_channels": "sum", "factor": 0.00001, "type": "local"}}},
                'use_pod_schedule': True,
            }





        @classmethod
        def from_config(cls, cfg):
            backbone = build_backbone(cfg)
            sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

            # Loss parameters:
            deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
            no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

            # loss weights
            class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
            dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
            mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

            # building criterion
            matcher = HungarianMatcher(
                cost_class=class_weight,
                cost_mask=mask_weight,
                cost_dice=dice_weight,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            )

            weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

            if deep_supervision:
                dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
                aux_weight_dict = {}
                for i in range(dec_layers - 1):
                    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
                weight_dict.update(aux_weight_dict)

            losses = ["labels", "masks"]

            dataset, seg, task, step = cfg.DATASETS.TRAIN[0].split('_')

            if dataset == 'ade20k':
                if seg == 'incrementalins':
                    from mask2former.data.datasets.register_ade20k_incremental_instance import tasks_ade_ins
                    training_classes = tasks_ade_ins[task][int(step)]
                else:
                    from mask2former.data.datasets.register_ade20k_incremental import tasks_ade
                    training_classes = tasks_ade[task][int(step)]
                cr_num_classes = len(training_classes)
            elif dataset == 'voc':
                from mask2former.data.datasets.register_voc_incremental import tasks_voc
                training_classes = tasks_voc[task][int(step)]
                cr_num_classes = len(training_classes)
            elif dataset == 'coco':
                from mask2former.data.datasets.register_coco_incremental_instance import tasks_coco_ins
                training_classes = tasks_coco_ins[task][int(step)]
                cr_num_classes = len(training_classes)
            else:
                print(f'no dataset {dataset}')
                raise ValueError('No dataset')

            criterion = SetCriterion(
                cr_num_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight,
                losses=losses,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
                oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
                importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            )

            return {
                "cfg": cfg,
                "name": cfg.DATASETS.TRAIN[0],
                "backbone": backbone,
                "sem_seg_head": sem_seg_head,
                "criterion": criterion,
                "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
                "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
                "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
                "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
                "sem_seg_postprocess_before_inference": (
                        cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                        or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                        or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
                ),
                "pixel_mean": cfg.MODEL.PIXEL_MEAN,
                "pixel_std": cfg.MODEL.PIXEL_STD,
                # inference
                "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
                "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
                "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
                "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            }

        @property
        def device(self):
            return self.pixel_mean.device


        def losskd(self, new_features, old_features):
            losskd = dict()

            # Prediction KD
            predictions_classs = old_features['predictions_class']
            ppredictions_classs = new_features['predictions_class']
            predictions_masks = old_features['predictions_mask']
            ppredictions_masks = new_features['predictions_mask']
            nl = [9]
            nk = predictions_classs[0].keys()
            for l in nl:
                total_nq = sum([ppredictions_classs[l][k].shape[1] for k in nk])
                losskd[f'loss_kd_class{l}'] = 0
                losskd[f'loss_kd_mask{l}'] = 0
                for k in nk:
                    pred_class = ppredictions_classs[l][k]
                    gt_class = predictions_classs[l][k]

                    b, nq, c = pred_class.shape

                    pred = pred_class.reshape(-1, c).softmax(-1)
                    gt = gt_class.reshape(-1, c).softmax(-1)
                    # 1000 for 11 task, 300 for 2 or 3 task
                    step_loss_kd_class = (F.kl_div(pred.log(), gt, reduction='sum') / (b * nq) * 300) * nq / total_nq
                    losskd[f'loss_kd_class{l}'] += step_loss_kd_class

                    pred_mask = ppredictions_masks[l][k]
                    gt_mask = predictions_masks[l][k]

                    b, nq, h, w = pred_mask.shape
                    pred = pred_mask.reshape(-1, h, w)
                    gt = (gt_mask.reshape(-1, h, w) > 0).to(torch.int64)
                    step_loss_kd_mask = (self.dice_loss(pred, gt) * 300) * nq / total_nq
                    losskd[f'loss_kd_mask{l}'] += step_loss_kd_mask




            pod_old_features = []
            for k, v in old_features['res_features'].items():
                pod_old_features.append(v)
            pod_old_features.append(old_features['mask_features'])
            pod_new_features = []
            for k, v in new_features['res_features'].items():
                pod_new_features.append(v)
            pod_new_features.append(new_features['mask_features'])


            losskd[f'loss_kd_lpod'] = features_distillation(pod_old_features, pod_new_features, **self.params)


            return losskd

        def detach_features(self, features):
            if isinstance(features, torch.Tensor):
                return features.detach()
            elif isinstance(features, dict):
                new_dict = dict()
                for k, v in features.items():
                    new_dict[k] = self.detach_features(v)
                return new_dict
            elif isinstance(features, list):
                return [self.detach_features(el) for el in features]
            else:
                raise ValueError("unknow type")

        def forward(self, batched_inputs, kd=False):
            if kd:
                return self.forward_old(batched_inputs)
            else:
                return self.forward_new(batched_inputs)

        def forward_old(self, batched_inputs):
            # forward path of the model
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            features = self.backbone(images.tensor)

            outputs = self.sem_seg_head(features)

            # semantic predictions
            sem_logits = self.semantic_inference(outputs['pred_logits'], outputs['pred_masks'])
            outputs['kd_features']['predictions_sem'] = sem_logits

            # return the kd features
            kd_features = outputs['kd_features']
            kd_features['res_features'] = features
            return self.detach_features(kd_features)

        def forward_new(self, batched_inputs):
            # forward path of the model
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            features = self.backbone(images.tensor)

            outputs = self.sem_seg_head(features)

            # semantic predictions
            sem_logits = self.semantic_inference(outputs['pred_logits'], outputs['pred_masks'])
            outputs['kd_features']['predictions_sem'] = sem_logits

            if self.training:
                # mask classification target
                if "instances" in batched_inputs[0]:
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                    gt_ignores = [x["ignore"].to(self.device) for x in batched_inputs]
                    targets = self.prepare_targets(gt_instances, gt_ignores, images)
                else:
                    targets = None

                losses = dict()

                # cal the kd loss
                if self.training and self.model_old is not None:
                    old_features = self.model_old['model_old'](batched_inputs, kd=True)
                    new_features = outputs['kd_features']
                    new_features['res_features'] = features
                    losses.update(self.losskd(new_features, old_features))

                if 'kd_features' in outputs:
                    outputs.pop('kd_features')

                # bipartite matching-based loss
                losses.update(self.criterion(outputs, targets))

                for k in list(losses.keys()):
                    if k in self.criterion.weight_dict:
                        losses[k] *= self.criterion.weight_dict[k]
                    else:
                        # remove this loss if not specified in `weight_dict`
                        continue
                        # losses.pop(k)
                return losses
            else:
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )

                del outputs

                processed_results = []
                for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                        mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    processed_results.append({})

                    if self.sem_seg_postprocess_before_inference:
                        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            mask_pred_result, image_size, height, width
                        )
                        mask_cls_result = mask_cls_result.to(mask_pred_result)

                    # semantic segmentation inference
                    if self.semantic_on:
                        r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                        if not self.sem_seg_postprocess_before_inference:
                            r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                        processed_results[-1]["sem_seg"] = r

                    # panoptic segmentation inference
                    if self.panoptic_on:
                        panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["panoptic_seg"] = panoptic_r

                    # instance segmentation inference
                    if self.instance_on:
                        instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["instances"] = instance_r

                return processed_results

        def prepare_targets(self, targets, gt_ignores, images):
            h_pad, w_pad = images.tensor.shape[-2:]
            new_targets = []
            for targets_per_image, gt_ignore in zip(targets, gt_ignores):
                # pad gt
                gt_masks = targets_per_image.gt_masks
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype,
                                           device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
                new_targets.append(
                    {
                        "labels": targets_per_image.gt_classes,
                        "masks": padded_masks,
                        "gt_ignore": gt_ignore,
                    }
                )
            return new_targets

        def semantic_inference(self, mask_cls, mask_pred):

            mask_cls = F.softmax(mask_cls, dim=-1)[..., 1:]  # the first class is the background
            mask_pred = mask_pred.sigmoid()
            if len(mask_cls.shape)==2:
                semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            else:
                semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
            return semseg

        def panoptic_inference(self, mask_cls, mask_pred):
            scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
            mask_pred = mask_pred.sigmoid()

            keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[keep]
            cur_mask_cls = mask_cls[keep]
            cur_mask_cls = cur_mask_cls[:, :-1]

            cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

            h, w = cur_masks.shape[-2:]
            panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
            segments_info = []

            current_segment_id = 0

            if cur_masks.shape[0] == 0:
                # We didn't detect any mask :(
                return panoptic_seg, segments_info
            else:
                # take argmax
                cur_mask_ids = cur_prob_masks.argmax(0)
                stuff_memory_list = {}
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()
                    isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                    mask_area = (cur_mask_ids == k).sum().item()
                    original_area = (cur_masks[k] >= 0.5).sum().item()
                    mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        if mask_area / original_area < self.overlap_threshold:
                            continue

                        # merge stuff regions
                        if not isthing:
                            if int(pred_class) in stuff_memory_list.keys():
                                panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                                continue
                            else:
                                stuff_memory_list[int(pred_class)] = current_segment_id + 1

                        current_segment_id += 1
                        panoptic_seg[mask] = current_segment_id

                        segments_info.append(
                            {
                                "id": current_segment_id,
                                "isthing": bool(isthing),
                                "category_id": int(pred_class),
                            }
                        )

                return panoptic_seg, segments_info

        def instance_inference(self, mask_cls, mask_pred):
            # mask_pred is already processed to have the same shape as original input
            image_size = mask_pred.shape[-2:]

            # [Q, K]
            # scores = F.softmax(mask_cls, dim=-1)[:, :-1]
            scores = F.softmax(mask_cls, dim=-1)[:, 1:]
            num_queries, num_classes = scores.shape
            labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
            # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
            labels_per_image = labels[topk_indices]

            topk_indices = topk_indices // num_classes
            # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
            mask_pred = mask_pred[topk_indices]

            # if this is panoptic segmentation, we only keep the "thing" classes
            if self.panoptic_on:
                keep = torch.zeros_like(scores_per_image).bool()
                for i, lab in enumerate(labels_per_image):
                    keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]
                mask_pred = mask_pred[keep]

            result = Instances(image_size)
            # mask (before sigmoid)
            result.pred_masks = (mask_pred > 0).float()
            result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
            # Uncomment the following to get boxes from masks (this is slow)
            # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

            # calculate average mask prob
            mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
                    result.pred_masks.flatten(1).sum(1) + 1e-6)
            result.scores = scores_per_image * mask_scores_per_image
            result.pred_classes = labels_per_image
            return result

elif version == 'pod_mm':
    # query freeze : True
    # query kd : True
    # prediction kd : True
    # query group attention mask : True


    class SoftDiceLoss(nn.Module):
        def __init__(self, weight=None, size_average=True):
            super(SoftDiceLoss, self).__init__()

        def forward(self, logits, targets):
            bs = targets.size(0)
            smooth = 1.0

            probs = F.sigmoid(logits)
            m1 = probs.view(bs, -1)
            m2 = targets.view(bs, -1)
            intersection = (m1 * m2)

            score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
            score = 1 - score.sum() / bs
            return score


    @META_ARCH_REGISTRY.register()
    class IncrementalMaskFormer(nn.Module):
        """
        Main class for mask classification semantic segmentation architectures.
        """

        @configurable
        def __init__(
                self,
                *,
                cfg: None,
                name: str,
                backbone: Backbone,
                sem_seg_head: nn.Module,
                criterion: nn.Module,
                num_queries: int,
                object_mask_threshold: float,
                overlap_threshold: float,
                metadata,
                size_divisibility: int,
                sem_seg_postprocess_before_inference: bool,
                pixel_mean: Tuple[float],
                pixel_std: Tuple[float],
                # inference
                semantic_on: bool,
                panoptic_on: bool,
                instance_on: bool,
                test_topk_per_image: int,
        ):
            """
            Args:
                backbone: a backbone module, must follow detectron2's backbone interface
                sem_seg_head: a module that predicts semantic segmentation from backbone features
                criterion: a module that defines the loss
                num_queries: int, number of queries
                object_mask_threshold: float, threshold to filter query based on classification score
                    for panoptic segmentation inference
                overlap_threshold: overlap threshold used in general inference for panoptic segmentation
                metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                    segmentation inference
                size_divisibility: Some backbones require the input height and width to be divisible by a
                    specific integer. We can use this to override such requirement.
                sem_seg_postprocess_before_inference: whether to resize the prediction back
                    to original input size before semantic segmentation inference or after.
                    For high-resolution dataset like Mapillary, resizing predictions before
                    inference will cause OOM error.
                pixel_mean, pixel_std: list or tuple with #channels element, representing
                    the per-channel mean and std to be used to normalize the input image
                semantic_on: bool, whether to output semantic segmentation prediction
                instance_on: bool, whether to output instance segmentation prediction
                panoptic_on: bool, whether to output panoptic segmentation prediction
                test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
            """
            super().__init__()
            self.cfg = copy.deepcopy(cfg)
            dataset, _, task, step = name.split('_')
            self.dataset = dataset
            self.task = task
            self.step = int(step)

            sem_seg_head.predictor.task = self.task
            sem_seg_head.predictor.step = self.step


            # freeze the old queries
            self.tmodel = None
            if self.step > 0:
                sem_seg_head.predictor.query_feat.weight.requires_grad = False
                for s in range(self.step - 1):
                    sem_seg_head.predictor.incre_query_feat_list[s].weight.requires_grad = False

                # get the teacher model
                if not self.cfg.MODEL.MODEL_OLD:
                    # load step 0 teacher model
                    model_base_cfg = copy.deepcopy(cfg)
                    model_base_cfg.defrost()
                    model_base_cfg.MODEL.MODEL_OLD = True
                    weights = model_base_cfg.MODEL.WEIGHTS
                    weights = os.path.join('/'.join(weights.split('/')[:-2]),'step0/cur.pth')
                    model_base_cfg.MODEL.WEIGHTS = weights

                    model_base = build_model(model_base_cfg)
                    model_base.step = 0
                    model_base.sem_seg_head.predictor.step = 0
                    DetectionCheckpointer(model_base, save_dir='./cache/debug').resume_or_load(
                        model_base_cfg.MODEL.WEIGHTS
                    )
                    model_base.eval()

                    # load step t-1 teacher model
                    model_old_cfg = copy.deepcopy(cfg)
                    model_old_cfg.defrost()
                    model_old_cfg.MODEL.MODEL_OLD = True

                    model_old = build_model(model_old_cfg)
                    model_old.step = self.step - 1
                    model_old.sem_seg_head.predictor.step = self.step - 1
                    DetectionCheckpointer(model_old, save_dir='./cache/debug').resume_or_load(
                        model_old_cfg.MODEL.WEIGHTS
                    )
                    model_old.eval()
                    self.tmodel = {'model_base': model_base, 'model_old': model_old}
                else:
                    self.tmodel = None


            self.backbone = backbone
            self.sem_seg_head = sem_seg_head
            self.criterion = criterion
            self.num_queries = num_queries
            self.overlap_threshold = overlap_threshold
            self.object_mask_threshold = object_mask_threshold
            self.metadata = metadata
            if size_divisibility < 0:
                # use backbone size_divisibility if not set
                size_divisibility = self.backbone.size_divisibility
            self.size_divisibility = size_divisibility
            self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
            self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

            # additional args
            self.semantic_on = semantic_on
            self.instance_on = instance_on
            self.panoptic_on = panoptic_on
            self.test_topk_per_image = test_topk_per_image

            if not self.semantic_on:
                assert self.sem_seg_postprocess_before_inference

            self.dice_loss = SoftDiceLoss()

            self.params = {
                'collapse_channels': 'local',
                'pod_apply': 'all',
                'pod_factor': 1.,
                'prepro': 'pow',
                'spp_scales': [1, 2, 4],
                'pod_options': {"switch": {"after": {"extra_channels": "sum", "factor": 0.00001, "type": "local"}}},
                'use_pod_schedule': True,
            }





        @classmethod
        def from_config(cls, cfg):
            backbone = build_backbone(cfg)
            sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

            # Loss parameters:
            deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
            no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

            # loss weights
            class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
            dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
            mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

            # building criterion
            matcher = HungarianMatcher(
                cost_class=class_weight,
                cost_mask=mask_weight,
                cost_dice=dice_weight,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            )

            weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

            if deep_supervision:
                dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
                aux_weight_dict = {}
                for i in range(dec_layers - 1):
                    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
                weight_dict.update(aux_weight_dict)

            losses = ["labels", "masks"]

            dataset, seg, task, step = cfg.DATASETS.TRAIN[0].split('_')

            if dataset == 'ade20k':
                if seg == 'incrementalins':
                    from mask2former.data.datasets.register_ade20k_incremental_instance import tasks_ade_ins
                    training_classes = tasks_ade_ins[task][int(step)]
                else:
                    from mask2former.data.datasets.register_ade20k_incremental import tasks_ade
                    training_classes = tasks_ade[task][int(step)]
                cr_num_classes = len(training_classes)
            elif dataset == 'voc':
                from mask2former.data.datasets.register_voc_incremental import tasks_voc
                training_classes = tasks_voc[task][int(step)]
                cr_num_classes = len(training_classes)
            elif dataset == 'coco':
                from mask2former.data.datasets.register_coco_incremental_instance import tasks_coco_ins
                training_classes = tasks_coco_ins[task][int(step)]
                cr_num_classes = len(training_classes)
            else:
                print(f'no dataset {dataset}')
                raise ValueError('No dataset')

            criterion = SetCriterion(
                cr_num_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight,
                losses=losses,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
                oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
                importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            )

            return {
                "cfg": cfg,
                "name": cfg.DATASETS.TRAIN[0],
                "backbone": backbone,
                "sem_seg_head": sem_seg_head,
                "criterion": criterion,
                "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
                "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
                "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
                "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
                "sem_seg_postprocess_before_inference": (
                        cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                        or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                        or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
                ),
                "pixel_mean": cfg.MODEL.PIXEL_MEAN,
                "pixel_std": cfg.MODEL.PIXEL_STD,
                # inference
                "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
                "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
                "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
                "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            }

        @property
        def device(self):
            return self.pixel_mean.device

        def losskd(self, new_features, old_features):
            losskd = dict()

            wpred = [300,150]

            # Prediction KD
            predictions_classs = old_features['predictions_class']
            ppredictions_classs = new_features['predictions_class']
            predictions_masks = old_features['predictions_mask']
            ppredictions_masks = new_features['predictions_mask']
            nl = [9]
            nk = predictions_classs[0].keys()
            for l in nl:
                total_nq = sum([ppredictions_classs[l][k].shape[1] for k in nk])
                losskd[f'loss_kd_class{l}'] = 0
                losskd[f'loss_kd_mask{l}'] = 0
                for k in nk:
                    pred_class = ppredictions_classs[l][k]
                    gt_class = predictions_classs[l][k]

                    b, nq, c = pred_class.shape

                    pred = pred_class.reshape(-1, c).softmax(-1)
                    gt = gt_class.reshape(-1, c).softmax(-1)
                    # 1000 for 11 task, 300 for 2 or 3 task
                    step_loss_kd_class = (F.kl_div(pred.log(), gt, reduction='sum') / (b * nq) * wpred[0]) * nq / total_nq
                    losskd[f'loss_kd_class{l}'] += step_loss_kd_class

                    pred_mask = ppredictions_masks[l][k]
                    gt_mask = predictions_masks[l][k]

                    b, nq, h, w = pred_mask.shape
                    pred = pred_mask.reshape(-1, h, w)
                    gt = (gt_mask.reshape(-1, h, w) > 0).to(torch.int64)
                    step_loss_kd_mask = (self.dice_loss(pred, gt) * wpred[1]) * nq / total_nq
                    losskd[f'loss_kd_mask{l}'] += step_loss_kd_mask

            wpod = [10,1000,1]

            # ResNet features KD
            pod_old_features = []
            pod_new_features = []
            for k, v in old_features['res_features'].items():
                pod_old_features.append(v)

            for k, v in new_features['res_features'].items():
                pod_new_features.append(v)
            losskd[f'loss_kd_pod_res'] = features_distillation(pod_old_features, pod_new_features, **self.params)*wpod[0]

            # mask features KD
            pod_old_features = []
            pod_new_features = []
            pod_old_features.append(old_features['mask_features'])
            pod_new_features.append(new_features['mask_features'])
            losskd[f'loss_kd_pod_mask'] = features_distillation(pod_old_features, pod_new_features, **self.params)*wpod[1]

            # query KD
            pod_old_features = []
            pod_new_features = []
            kd_query_steps = old_features['predictions_query'][0].keys()
            for k1 in kd_query_steps:
                for old_feature, new_feature in zip(old_features['predictions_query'], new_features['predictions_query']):
                    pod_old_features.append(old_feature[k1].reshape(-1,256)[...,None,None])
                    pod_new_features.append(new_feature[k1].reshape(-1,256)[...,None,None])
            losskd[f'loss_kd_pod_query'] = features_distillation(pod_old_features, pod_new_features, **self.params)*wpod[2]


            return losskd

        def detach_features(self, features):
            if isinstance(features, torch.Tensor):
                return features.detach()
            elif isinstance(features, dict):
                new_dict = dict()
                for k, v in features.items():
                    new_dict[k] = self.detach_features(v)
                return new_dict
            elif isinstance(features, list):
                return [self.detach_features(el) for el in features]
            else:
                raise ValueError("unknow type")

        def unify_features(self, base_features, old_features):
            for old_feat, base_feat in zip(old_features['predictions_class'],base_features['predictions_class']):
                old_feat['step0'] = base_feat['step0']
            for old_feat, base_feat in zip(old_features['predictions_mask'],base_features['predictions_mask']):
                old_feat['step0'] = base_feat['step0']
            for old_feat, base_feat in zip(old_features['predictions_query'],base_features['predictions_query']):
                old_feat['step0'] = base_feat['step0']
            return old_features

        def forward(self, batched_inputs, kd=False):
            if kd:
                return self.forward_old(batched_inputs)
            else:
                return self.forward_new(batched_inputs)

        def forward_old(self, batched_inputs):
            # forward path of the model
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            features = self.backbone(images.tensor)

            outputs = self.sem_seg_head(features)

            # semantic predictions
            sem_logits = self.semantic_inference(outputs['pred_logits'], outputs['pred_masks'])
            outputs['kd_features']['predictions_sem'] = sem_logits

            # return the kd features
            kd_features = outputs['kd_features']
            kd_features['res_features'] = features
            return self.detach_features(kd_features)

        def forward_new(self, batched_inputs):
            # forward path of the model
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            features = self.backbone(images.tensor)

            outputs = self.sem_seg_head(features)

            # semantic predictions
            sem_logits = self.semantic_inference(outputs['pred_logits'], outputs['pred_masks'])
            outputs['kd_features']['predictions_sem'] = sem_logits

            if self.training:
                # mask classification target
                if "instances" in batched_inputs[0]:
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                    gt_ignores = [x["ignore"].to(self.device) for x in batched_inputs]
                    targets = self.prepare_targets(gt_instances, gt_ignores, images)
                else:
                    targets = None

                losses = dict()

                # cal the kd loss
                if self.training and self.tmodel is not None:
                    base_features = self.tmodel['model_base'](batched_inputs, kd=True)
                    old_features = self.tmodel['model_old'](batched_inputs, kd=True)
                    old_features = self.unify_features(base_features, old_features)
                    new_features = outputs['kd_features']
                    new_features['res_features'] = features
                    losses.update(self.losskd(new_features, old_features))

                if 'kd_features' in outputs:
                    outputs.pop('kd_features')

                # bipartite matching-based loss
                losses.update(self.criterion(outputs, targets))

                for k in list(losses.keys()):
                    if k in self.criterion.weight_dict:
                        losses[k] *= self.criterion.weight_dict[k]
                    else:
                        # remove this loss if not specified in `weight_dict`
                        continue
                        # losses.pop(k)
                return losses
            else:
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )

                del outputs

                processed_results = []
                for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                        mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    processed_results.append({})

                    if self.sem_seg_postprocess_before_inference:
                        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            mask_pred_result, image_size, height, width
                        )
                        mask_cls_result = mask_cls_result.to(mask_pred_result)

                    # semantic segmentation inference
                    if self.semantic_on:
                        r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                        if not self.sem_seg_postprocess_before_inference:
                            r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                        processed_results[-1]["sem_seg"] = r

                    # panoptic segmentation inference
                    if self.panoptic_on:
                        panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["panoptic_seg"] = panoptic_r

                    # instance segmentation inference
                    if self.instance_on:
                        instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["instances"] = instance_r

                return processed_results

        def prepare_targets(self, targets, gt_ignores, images):
            h_pad, w_pad = images.tensor.shape[-2:]
            new_targets = []
            for targets_per_image, gt_ignore in zip(targets, gt_ignores):
                # pad gt
                gt_masks = targets_per_image.gt_masks
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype,
                                           device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
                new_targets.append(
                    {
                        "labels": targets_per_image.gt_classes,
                        "masks": padded_masks,
                        "gt_ignore": gt_ignore,
                    }
                )
            return new_targets

        def semantic_inference(self, mask_cls, mask_pred):

            mask_cls = F.softmax(mask_cls, dim=-1)[..., 1:]  # the first class is the background
            mask_pred = mask_pred.sigmoid()
            if len(mask_cls.shape)==2:
                semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            else:
                semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
            return semseg

        def panoptic_inference(self, mask_cls, mask_pred):
            scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
            mask_pred = mask_pred.sigmoid()

            keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[keep]
            cur_mask_cls = mask_cls[keep]
            cur_mask_cls = cur_mask_cls[:, :-1]

            cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

            h, w = cur_masks.shape[-2:]
            panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
            segments_info = []

            current_segment_id = 0

            if cur_masks.shape[0] == 0:
                # We didn't detect any mask :(
                return panoptic_seg, segments_info
            else:
                # take argmax
                cur_mask_ids = cur_prob_masks.argmax(0)
                stuff_memory_list = {}
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()
                    isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                    mask_area = (cur_mask_ids == k).sum().item()
                    original_area = (cur_masks[k] >= 0.5).sum().item()
                    mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        if mask_area / original_area < self.overlap_threshold:
                            continue

                        # merge stuff regions
                        if not isthing:
                            if int(pred_class) in stuff_memory_list.keys():
                                panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                                continue
                            else:
                                stuff_memory_list[int(pred_class)] = current_segment_id + 1

                        current_segment_id += 1
                        panoptic_seg[mask] = current_segment_id

                        segments_info.append(
                            {
                                "id": current_segment_id,
                                "isthing": bool(isthing),
                                "category_id": int(pred_class),
                            }
                        )

                return panoptic_seg, segments_info

        def instance_inference(self, mask_cls, mask_pred):
            # mask_pred is already processed to have the same shape as original input
            image_size = mask_pred.shape[-2:]

            # [Q, K]
            # scores = F.softmax(mask_cls, dim=-1)[:, :-1]
            scores = F.softmax(mask_cls, dim=-1)[:, 1:]
            num_queries, num_classes = scores.shape
            labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
            # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
            labels_per_image = labels[topk_indices]

            topk_indices = topk_indices // num_classes
            # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
            mask_pred = mask_pred[topk_indices]

            # if this is panoptic segmentation, we only keep the "thing" classes
            if self.panoptic_on:
                keep = torch.zeros_like(scores_per_image).bool()
                for i, lab in enumerate(labels_per_image):
                    keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]
                mask_pred = mask_pred[keep]

            result = Instances(image_size)
            # mask (before sigmoid)
            result.pred_masks = (mask_pred > 0).float()
            result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
            # Uncomment the following to get boxes from masks (this is slow)
            # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

            # calculate average mask prob
            mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
                    result.pred_masks.flatten(1).sum(1) + 1e-6)
            result.scores = scores_per_image * mask_scores_per_image
            result.pred_classes = labels_per_image
            return result

# elif version == 'cisdq':
#     # query freeze : True
#     # query kd : True
#     # prediction kd : True
#     # query group attention mask : True
#
#     @META_ARCH_REGISTRY.register()
#     class IncrementalMaskFormer(nn.Module):
#         """
#         Main class for mask classification semantic segmentation architectures.
#         """
#
#         @configurable
#         def __init__(
#                 self,
#                 *,
#                 cfg: None,
#                 name: str,
#                 backbone: Backbone,
#                 sem_seg_head: nn.Module,
#                 criterion: nn.Module,
#                 num_queries: int,
#                 object_mask_threshold: float,
#                 overlap_threshold: float,
#                 metadata,
#                 size_divisibility: int,
#                 sem_seg_postprocess_before_inference: bool,
#                 pixel_mean: Tuple[float],
#                 pixel_std: Tuple[float],
#                 # inference
#                 semantic_on: bool,
#                 panoptic_on: bool,
#                 instance_on: bool,
#                 test_topk_per_image: int,
#         ):
#             """
#             Args:
#                 backbone: a backbone module, must follow detectron2's backbone interface
#                 sem_seg_head: a module that predicts semantic segmentation from backbone features
#                 criterion: a module that defines the loss
#                 num_queries: int, number of queries
#                 object_mask_threshold: float, threshold to filter query based on classification score
#                     for panoptic segmentation inference
#                 overlap_threshold: overlap threshold used in general inference for panoptic segmentation
#                 metadata: dataset meta, get `thing` and `stuff` category names for panoptic
#                     segmentation inference
#                 size_divisibility: Some backbones require the input height and width to be divisible by a
#                     specific integer. We can use this to override such requirement.
#                 sem_seg_postprocess_before_inference: whether to resize the prediction back
#                     to original input size before semantic segmentation inference or after.
#                     For high-resolution dataset like Mapillary, resizing predictions before
#                     inference will cause OOM error.
#                 pixel_mean, pixel_std: list or tuple with #channels element, representing
#                     the per-channel mean and std to be used to normalize the input image
#                 semantic_on: bool, whether to output semantic segmentation prediction
#                 instance_on: bool, whether to output instance segmentation prediction
#                 panoptic_on: bool, whether to output panoptic segmentation prediction
#                 test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
#             """
#             super().__init__()
#             self.cfg = copy.deepcopy(cfg)
#             dataset, _, task, step = name.split('_')
#             self.dataset = dataset
#             self.task = task
#             self.step = int(step)
#
#             sem_seg_head.predictor.task = self.task
#             sem_seg_head.predictor.step = self.step
#
#             # weighting schedule
#             if name.startswith('ade20k_incremental_100-50'):
#                 self.w = [300, 5, 10, 1000, 0.2, 1]
#             elif name.startswith('ade20k_incremental_50-50'):
#                 self.w = [300, 5, 10, 1000, 0.2, 1]
#             elif name.startswith('voc_incremental_15-5'):
#                 self.w = [300, 5, 10, 1000, 0.2, 1]
#             elif name.startswith('voc_incremental_19-1'):
#                 self.w = [300, 5, 10, 1000, 0.2, 1]
#             elif name.startswith('coco_incrementalins'):
#                 if '40-40' in name:
#                     self.w = [120, 2, 10, 1000, 1, 0.4]
#                 else:
#                     self.w = [300, 5, 10, 1000, 0.2, 0.1]
#             else:
#                 self.w = [300, 5, 100, 10000, 1, 0.1]
#
#
#             # freeze the old queries
#             self.tmodel = None
#             if self.step > 0:
#                 sem_seg_head.predictor.query_feat.weight.requires_grad = False
#                 for s in range(self.step - 1):
#                     sem_seg_head.predictor.incre_query_feat_list[s].weight.requires_grad = False
#
#                 # get the teacher model
#                 if not self.cfg.MODEL.MODEL_OLD:
#                     # load step 0 teacher model
#                     # model_base_cfg = copy.deepcopy(cfg)
#                     # model_base_cfg.defrost()
#                     # model_base_cfg.MODEL.MODEL_OLD = True
#                     # weights = model_base_cfg.MODEL.WEIGHTS
#                     # weights = os.path.join('/'.join(weights.split('/')[:-2]),'step0/cur.pth')
#                     # model_base_cfg.MODEL.WEIGHTS = weights
#                     #
#                     # model_base = build_model(model_base_cfg)
#                     # model_base.step = 0
#                     # model_base.sem_seg_head.predictor.step = 0
#                     # DetectionCheckpointer(model_base, save_dir='./cache/debug').resume_or_load(
#                     #     model_base_cfg.MODEL.WEIGHTS
#                     # )
#                     # model_base.eval()
#
#                     # load step t-1 teacher model
#                     model_old_cfg = copy.deepcopy(cfg)
#                     model_old_cfg.defrost()
#                     model_old_cfg.MODEL.MODEL_OLD = True
#
#                     model_old = build_model(model_old_cfg)
#                     model_old.step = self.step - 1
#                     model_old.sem_seg_head.predictor.step = self.step - 1
#                     DetectionCheckpointer(model_old, save_dir='./cache/debug').resume_or_load(
#                         model_old_cfg.MODEL.WEIGHTS
#                     )
#                     model_old.eval()
#                     self.tmodel = {'model_base': model_old, 'model_old': model_old}
#                 else:
#                     self.tmodel = None
#
#
#             self.backbone = backbone
#             self.sem_seg_head = sem_seg_head
#             self.criterion = criterion
#             self.num_queries = num_queries
#             self.overlap_threshold = overlap_threshold
#             self.object_mask_threshold = object_mask_threshold
#             self.metadata = metadata
#             if size_divisibility < 0:
#                 # use backbone size_divisibility if not set
#                 size_divisibility = self.backbone.size_divisibility
#             self.size_divisibility = size_divisibility
#             self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
#             self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
#             self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
#
#             # additional args
#             self.semantic_on = semantic_on
#             self.instance_on = instance_on
#             self.panoptic_on = panoptic_on
#             self.test_topk_per_image = test_topk_per_image
#
#             if not self.semantic_on:
#                 assert self.sem_seg_postprocess_before_inference
#
#             self.params = {
#                 'collapse_channels': 'local',
#                 'pod_apply': 'all',
#                 'pod_factor': 1.,
#                 'prepro': 'pow',
#                 'spp_scales': [1, 2, 4],
#                 'pod_options': {"switch": {"after": {"extra_channels": "sum", "factor": 0.00001, "type": "local"}}},
#                 'use_pod_schedule': True,
#             }
#
#             self.num_base_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
#             self.num_novel_classes = cfg.MODEL.MASK_FORMER.NUM_INCRE_CLASSES
#
#
#
#
#
#         @classmethod
#         def from_config(cls, cfg):
#             backbone = build_backbone(cfg)
#             sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
#
#             # Loss parameters:
#             deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
#             no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
#
#             # loss weights
#             class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
#             dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
#             mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
#
#             # building criterion
#             matcher = HungarianMatcher(
#                 cost_class=class_weight,
#                 cost_mask=mask_weight,
#                 cost_dice=dice_weight,
#                 num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
#             )
#
#             weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
#
#             if deep_supervision:
#                 dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
#                 aux_weight_dict = {}
#                 for i in range(dec_layers - 1):
#                     aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
#                 weight_dict.update(aux_weight_dict)
#
#             losses = ["labels", "masks"]
#
#             num_base_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
#             num_novel_classes = cfg.MODEL.MASK_FORMER.NUM_INCRE_CLASSES
#
#             criterion = nn.ModuleDict()
#
#             base_criterion = SetCriterion(
#                 num_base_classes,
#                 matcher=matcher,
#                 weight_dict=weight_dict,
#                 eos_coef=no_object_weight,
#                 losses=losses,
#                 num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
#                 oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
#                 importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
#             )
#
#             novel_criterion = SetCriterion(
#                 num_novel_classes,
#                 matcher=matcher,
#                 weight_dict=weight_dict,
#                 eos_coef=no_object_weight,
#                 losses=losses,
#                 num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
#                 oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
#                 importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
#             )
#
#             criterion['base_criterion'] = base_criterion
#             criterion['novel_criterion'] = novel_criterion
#
#             return {
#                 "cfg": cfg,
#                 "name": cfg.DATASETS.TRAIN[0],
#                 "backbone": backbone,
#                 "sem_seg_head": sem_seg_head,
#                 "criterion": criterion,
#                 "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
#                 "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
#                 "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
#                 "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
#                 "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
#                 "sem_seg_postprocess_before_inference": (
#                         cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
#                         or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
#                         or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
#                 ),
#                 "pixel_mean": cfg.MODEL.PIXEL_MEAN,
#                 "pixel_std": cfg.MODEL.PIXEL_STD,
#                 # inference
#                 "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
#                 "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
#                 "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
#                 "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
#             }
#
#         @property
#         def device(self):
#             return self.pixel_mean.device
#
#         def entropy(self,probabilities):
#             """Computes the entropy per pixel.
#
#             # References:
#                 * ESL: Entropy-guided Self-supervised Learning for Domain Adaptation in Semantic Segmentation
#                   Saporta et al.
#                   CVPR Workshop 2020
#
#             :param probabilities: Tensor of shape (b, c, w, h).
#             :return: One entropy per pixel, shape (b, w, h)
#             """
#             factor = 1 / math.log(probabilities.shape[1] + 1e-8)
#             return -factor * torch.mean(probabilities * torch.log(probabilities + 1e-8), dim=1)
#
#
#         def losskd(self, new_features, old_features, batched_inputs, targets):
#             losskd = dict()
#
#             kd_groups = old_features['predictions_class'][0].keys()
#
#             # background masks
#             bg_masks = []
#             for target in targets:
#                 mask = target['labels'] == 0
#                 bg_mask = target['masks'][mask]
#                 if len(bg_mask)==0:
#                     bg_mask = torch.zeros(target['masks'].shape[-2:]).to(target['masks'])
#                 else:
#                     bg_mask = bg_mask[0]
#                 bg_masks.append(bg_mask)
#             h, w = bg_masks[0].shape
#
#
#             for group in kd_groups:
#                 # format gt
#                 gt_logits = old_features['predictions_class'][-1][group]
#                 gt_masks = old_features['predictions_mask'][-1][group]
#                 pesudo_targets = []
#                 for mask_cls_result, mask_pred_result, bg_mask in zip(gt_logits, gt_masks, bg_masks):
#                     # if self.instance_on:
#                     instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
#
#                     # prediction of the previous model
#                     score_thr = 0.1
#                     pesudo_label = {'labels': [], 'masks': [], 'gt_ignore': None}
#                     scores = instance_r.scores
#                     scores_mask = scores > score_thr
#                     pred_masks = F.interpolate(instance_r.pred_masks[None],(h,w),mode='nearest')[0]
#                     pred_masks = pred_masks[scores_mask]
#                     pred_classes = instance_r.pred_classes[scores_mask] + 1
#
#                     if group == 'step0':
#                         # merge background class for the base group
#                         non_bg_mask = pred_classes!=1
#                         non_bg_pred_classes = pred_classes[non_bg_mask]
#                         non_bg_pred_masks = pred_masks[non_bg_mask]
#                         if len(non_bg_pred_masks)>0:
#                             bg_mask = (bg_mask*~non_bg_pred_masks.sum(0).to(torch.bool)).to(torch.bool)
#                         bg_class = torch.tensor([1]).to(torch.int64).to(non_bg_pred_classes)
#                         pred_classes = torch.cat([bg_class,non_bg_pred_classes],0).to(torch.int64)
#                         pred_masks = torch.cat([bg_mask[None],non_bg_pred_masks],0).to(torch.bool)
#
#                     pesudo_label['labels'] = pred_classes.to(torch.int64)
#                     pesudo_label['masks'] = pred_masks.to(torch.bool)
#                     pesudo_targets.append(pesudo_label)
#
#                 # format pred
#                 outputs = dict()
#                 pred_logits = [el[group] for el in new_features['predictions_class']]
#                 pred_masks = [el[group] for el in new_features['predictions_mask']]
#                 outputs['pred_logits'] = pred_logits[-1]
#                 outputs['pred_masks'] = pred_masks[-1]
#                 aux_outputs = []
#                 for el1, el2 in zip(pred_logits[:-1],pred_masks[:-1]):
#                     aux_outputs.append({'pred_logits':el1, 'pred_masks':el2})
#                 outputs['aux_outputs'] = aux_outputs
#
#                 # set loss for a single group queries
#                 if group == 'step0':
#                     criterion = self.criterion['base_criterion']
#                     loss_g = criterion(outputs, pesudo_targets)
#                 else:
#                     criterion = self.criterion['novel_criterion']
#                     loss_g = criterion(outputs, pesudo_targets)
#
#                 for k in list(loss_g.keys()):
#                     if k in criterion.weight_dict:
#                         loss_g[k] *= criterion.weight_dict[k]
#                     else:
#                         continue
#                 losskd[f'loss_{group}'] = sum(loss_g.values())
#
#             # for k in list(losskd.keys()):
#             #     if 'step0' in k:
#             #         losskd[k] *= wpred[0]
#             #     else:
#             #         losskd[k] *= wpred[1]
#
#
#             # kernel = np.ones((10, 10), np.uint8)
#             # fg_masks = [(el['sem_seg']!=255).detach().cpu().numpy() for el in batched_inputs]
#             # bg_masks = []
#             # for mask in fg_masks:
#             #     bg_masks.append(mask.astype(np.float))
#             #     # bg_masks.append(cv2.dilate(mask.astype(np.float), kernel))
#             # bg_masks = torch.from_numpy(1-np.stack(bg_masks,0)[:,None]).to(torch.float)
#
#             wpod = [100,1000,1]
#
#             # ResNet features KD
#             pod_old_features = []
#             pod_new_features = []
#             for k, v in old_features['res_features'].items():
#                 _, _, h, w = v.shape
#                 # masks = F.interpolate(bg_masks,(h,w)).to(v)
#                 pod_old_features.append(v)
#
#             for k, v in new_features['res_features'].items():
#                 _, _, h, w = v.shape
#                 # masks = F.interpolate(bg_masks, (h, w)).to(v)
#                 pod_new_features.append(v)
#             losskd[f'loss_kd_pod_res'] = features_distillation(pod_old_features, pod_new_features, **self.params)*wpod[0]
#
#             # mask features KD
#             pod_old_features = []
#             pod_new_features = []
#             pod_old_features.append(old_features['mask_features'])
#             pod_new_features.append(new_features['mask_features'])
#             losskd[f'loss_kd_pod_mask'] = features_distillation(pod_old_features, pod_new_features, **self.params)*wpod[1]
#
#             # query KD
#             pod_old_features = []
#             pod_new_features = []
#             kd_query_steps = old_features['predictions_query'][0].keys()
#             for k1 in kd_query_steps:
#                 for old_feature, new_feature in zip(old_features['predictions_query'], new_features['predictions_query']):
#                     pod_old_features.append(old_feature[k1].reshape(-1,256)[...,None,None])
#                     pod_new_features.append(new_feature[k1].reshape(-1,256)[...,None,None])
#             losskd[f'loss_kd_pod_query'] = features_distillation(pod_old_features, pod_new_features, **self.params)*wpod[2]
#
#
#             return losskd
#
#         def detach_features(self, features):
#             if isinstance(features, torch.Tensor):
#                 return features.detach()
#             elif isinstance(features, dict):
#                 new_dict = dict()
#                 for k, v in features.items():
#                     new_dict[k] = self.detach_features(v)
#                 return new_dict
#             elif isinstance(features, list):
#                 return [self.detach_features(el) for el in features]
#             else:
#                 raise ValueError("unknow type")
#
#         def unify_features(self, base_features, old_features):
#             for old_feat, base_feat in zip(old_features['predictions_class'],base_features['predictions_class']):
#                 old_feat['step0'] = base_feat['step0']
#             for old_feat, base_feat in zip(old_features['predictions_mask'],base_features['predictions_mask']):
#                 old_feat['step0'] = base_feat['step0']
#             for old_feat, base_feat in zip(old_features['predictions_query'],base_features['predictions_query']):
#                 old_feat['step0'] = base_feat['step0']
#             return old_features
#
#         def forward(self, batched_inputs, kd=False):
#             if kd:
#                 return self.forward_old(batched_inputs)
#             else:
#                 return self.forward_new(batched_inputs)
#
#         def forward_old(self, batched_inputs):
#             # forward path of the model
#             images = [x["image"].to(self.device) for x in batched_inputs]
#             images = [(x - self.pixel_mean) / self.pixel_std for x in images]
#             images = ImageList.from_tensors(images, self.size_divisibility)
#
#             features = self.backbone(images.tensor)
#
#             outputs = self.sem_seg_head(features)
#
#             # semantic predictions
#             sem_logits = self.semantic_inference(outputs['pred_logits'], outputs['pred_masks'])
#             outputs['kd_features']['predictions_sem'] = sem_logits
#
#             # return the kd features
#             kd_features = outputs['kd_features']
#             kd_features['res_features'] = features
#             return self.detach_features(kd_features)
#
#         def forward_new(self, batched_inputs):
#             # forward path of the model
#             images = [x["image"].to(self.device) for x in batched_inputs]
#             images = [(x - self.pixel_mean) / self.pixel_std for x in images]
#             images = ImageList.from_tensors(images, self.size_divisibility)
#
#             features = self.backbone(images.tensor)
#
#             outputs = self.sem_seg_head(features)
#
#             # semantic predictions
#             sem_logits = self.semantic_inference(outputs['pred_logits'], outputs['pred_masks'])
#             outputs['kd_features']['predictions_sem'] = sem_logits
#
#             if self.training:
#                 # mask classification target
#                 if "instances" in batched_inputs[0]:
#                     gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
#                     gt_ignores = [x["ignore"].to(self.device) for x in batched_inputs]
#                     targets = self.prepare_targets(gt_instances, gt_ignores, images)
#                 else:
#                     targets = None
#
#                 losses = dict()
#
#                 # cal the kd loss
#                 if self.training and self.tmodel is not None:
#                     # base_features = self.tmodel['model_base'](batched_inputs, kd=True)
#                     old_features = self.tmodel['model_old'](batched_inputs, kd=True)
#                     # old_features = self.unify_features(base_features, old_features)
#                     new_features = outputs['kd_features']
#                     new_features['res_features'] = features
#                     losses.update(self.losskd(new_features, old_features, batched_inputs, targets))
#
#                     # mask the background class for the novel class training
#                     for target in targets:
#                         mask = target['labels']>0
#                         target['labels'] = target['labels'][mask]
#                         target['masks'] = target['masks'][mask]
#
#                 if 'kd_features' in outputs:
#                     outputs.pop('kd_features')
#
#                 # bipartite matching-based loss
#                 if self.step == 0:
#                     criterion = self.criterion['base_criterion']
#                 else:
#                     criterion = self.criterion['novel_criterion']
#                 loss_novel = criterion(outputs, targets)
#                 # losses.update(criterion(outputs, targets))
#
#                 for k in list(loss_novel.keys()):
#                     if k in criterion.weight_dict:
#                         loss_novel[k] *= criterion.weight_dict[k]
#                     else:
#                         continue
#                 loss_novel = sum(loss_novel.values())
#                 losses[f'loss_step{self.step}'] = loss_novel
#
#                 # weighting group loss by number of classes
#                 weight_sum = self.num_base_classes + self.step * self.num_novel_classes
#                 for i in range(self.step+1):
#                     key = f'loss_step{i}'
#                     if i == 0:
#                         losses[key] *= self.num_base_classes / weight_sum
#                     else:
#                         losses[key] *= self.num_novel_classes / weight_sum
#
#                 return losses
#             else:
#                 mask_cls_results = outputs["pred_logits"]
#                 mask_pred_results = outputs["pred_masks"]
#                 # upsample masks
#                 mask_pred_results = F.interpolate(
#                     mask_pred_results,
#                     size=(images.tensor.shape[-2], images.tensor.shape[-1]),
#                     mode="bilinear",
#                     align_corners=False,
#                 )
#
#                 del outputs
#
#                 processed_results = []
#                 for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
#                         mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
#                 ):
#                     height = input_per_image.get("height", image_size[0])
#                     width = input_per_image.get("width", image_size[1])
#                     processed_results.append({})
#
#                     if self.sem_seg_postprocess_before_inference:
#                         mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
#                             mask_pred_result, image_size, height, width
#                         )
#                         mask_cls_result = mask_cls_result.to(mask_pred_result)
#
#                     # semantic segmentation inference
#                     if self.semantic_on:
#                         r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
#                         if not self.sem_seg_postprocess_before_inference:
#                             r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
#                         processed_results[-1]["sem_seg"] = r
#
#                     # panoptic segmentation inference
#                     if self.panoptic_on:
#                         panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
#                         processed_results[-1]["panoptic_seg"] = panoptic_r
#
#                     # instance segmentation inference
#                     if self.instance_on:
#                         instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
#                         processed_results[-1]["instances"] = instance_r
#
#                 return processed_results
#
#         def prepare_targets(self, targets, gt_ignores, images):
#             h_pad, w_pad = images.tensor.shape[-2:]
#             new_targets = []
#             for targets_per_image, gt_ignore in zip(targets, gt_ignores):
#                 # pad gt
#                 gt_masks = targets_per_image.gt_masks
#                 padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype,
#                                            device=gt_masks.device)
#                 padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
#                 new_targets.append(
#                     {
#                         "labels": targets_per_image.gt_classes,
#                         "masks": padded_masks,
#                         "gt_ignore": gt_ignore,
#                     }
#                 )
#             return new_targets
#
#         def semantic_inference(self, mask_cls, mask_pred):
#             mask_cls = F.softmax(mask_cls, dim=-1)[..., 1:]  # the first class is the background
#             # mask_cls[...,0] = 0
#             mask_cls = F.softmax(mask_cls, dim=-1)
#             mask_pred = mask_pred.sigmoid()
#             if len(mask_cls.shape)==2:
#                 semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
#             else:
#                 semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
#             return semseg
#
#         def panoptic_inference(self, mask_cls, mask_pred):
#             scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
#             mask_pred = mask_pred.sigmoid()
#
#             keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
#             cur_scores = scores[keep]
#             cur_classes = labels[keep]
#             cur_masks = mask_pred[keep]
#             cur_mask_cls = mask_cls[keep]
#             cur_mask_cls = cur_mask_cls[:, :-1]
#
#             cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks
#
#             h, w = cur_masks.shape[-2:]
#             panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
#             segments_info = []
#
#             current_segment_id = 0
#
#             if cur_masks.shape[0] == 0:
#                 # We didn't detect any mask :(
#                 return panoptic_seg, segments_info
#             else:
#                 # take argmax
#                 cur_mask_ids = cur_prob_masks.argmax(0)
#                 stuff_memory_list = {}
#                 for k in range(cur_classes.shape[0]):
#                     pred_class = cur_classes[k].item()
#                     isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
#                     mask_area = (cur_mask_ids == k).sum().item()
#                     original_area = (cur_masks[k] >= 0.5).sum().item()
#                     mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)
#
#                     if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
#                         if mask_area / original_area < self.overlap_threshold:
#                             continue
#
#                         # merge stuff regions
#                         if not isthing:
#                             if int(pred_class) in stuff_memory_list.keys():
#                                 panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
#                                 continue
#                             else:
#                                 stuff_memory_list[int(pred_class)] = current_segment_id + 1
#
#                         current_segment_id += 1
#                         panoptic_seg[mask] = current_segment_id
#
#                         segments_info.append(
#                             {
#                                 "id": current_segment_id,
#                                 "isthing": bool(isthing),
#                                 "category_id": int(pred_class),
#                             }
#                         )
#
#                 return panoptic_seg, segments_info
#
#         def instance_inference(self, mask_cls, mask_pred):
#             # mask_pred is already processed to have the same shape as original input
#             image_size = mask_pred.shape[-2:]
#
#             # [Q, K]
#             # scores = F.softmax(mask_cls, dim=-1)[:, :-1]
#             scores = F.softmax(mask_cls, dim=-1)[:, 1:]
#             num_queries, num_classes = scores.shape
#             labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
#             # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
#             test_topk_per_image = min(len(scores),self.test_topk_per_image)
#             scores_per_image, topk_indices = scores.flatten(0, 1).topk(test_topk_per_image, sorted=False)
#             labels_per_image = labels[topk_indices]
#
#             topk_indices = topk_indices // num_classes
#             # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
#             mask_pred = mask_pred[topk_indices]
#
#             # if this is panoptic segmentation, we only keep the "thing" classes
#             if self.panoptic_on:
#                 keep = torch.zeros_like(scores_per_image).bool()
#                 for i, lab in enumerate(labels_per_image):
#                     keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()
#
#                 scores_per_image = scores_per_image[keep]
#                 labels_per_image = labels_per_image[keep]
#                 mask_pred = mask_pred[keep]
#
#             result = Instances(image_size)
#             # mask (before sigmoid)
#             result.pred_masks = (mask_pred > 0).float()
#             result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
#             # Uncomment the following to get boxes from masks (this is slow)
#             # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()
#
#             # calculate average mask prob
#             mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
#                     result.pred_masks.flatten(1).sum(1) + 1e-6)
#             result.scores = scores_per_image * mask_scores_per_image
#             result.pred_classes = labels_per_image
#             return result

elif version == 'cisdq':
    # query freeze : True
    # query kd : True
    # prediction kd : True
    # query group attention mask : True

    @META_ARCH_REGISTRY.register()
    class IncrementalMaskFormer(nn.Module):
        """
        Main class for mask classification semantic segmentation architectures.
        """

        @configurable
        def __init__(
                self,
                *,
                cfg: None,
                name: str,
                backbone: Backbone,
                sem_seg_head: nn.Module,
                criterion: nn.Module,
                num_queries: int,
                object_mask_threshold: float,
                overlap_threshold: float,
                metadata,
                size_divisibility: int,
                sem_seg_postprocess_before_inference: bool,
                pixel_mean: Tuple[float],
                pixel_std: Tuple[float],
                # inference
                semantic_on: bool,
                panoptic_on: bool,
                instance_on: bool,
                test_topk_per_image: int,
        ):
            """
            Args:
                backbone: a backbone module, must follow detectron2's backbone interface
                sem_seg_head: a module that predicts semantic segmentation from backbone features
                criterion: a module that defines the loss
                num_queries: int, number of queries
                object_mask_threshold: float, threshold to filter query based on classification score
                    for panoptic segmentation inference
                overlap_threshold: overlap threshold used in general inference for panoptic segmentation
                metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                    segmentation inference
                size_divisibility: Some backbones require the input height and width to be divisible by a
                    specific integer. We can use this to override such requirement.
                sem_seg_postprocess_before_inference: whether to resize the prediction back
                    to original input size before semantic segmentation inference or after.
                    For high-resolution dataset like Mapillary, resizing predictions before
                    inference will cause OOM error.
                pixel_mean, pixel_std: list or tuple with #channels element, representing
                    the per-channel mean and std to be used to normalize the input image
                semantic_on: bool, whether to output semantic segmentation prediction
                instance_on: bool, whether to output instance segmentation prediction
                panoptic_on: bool, whether to output panoptic segmentation prediction
                test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
            """
            super().__init__()
            self.cfg = copy.deepcopy(cfg)
            dataset, _, task, step = name.split('_')
            self.dataset = dataset
            self.task = task
            self.step = int(step)

            sem_seg_head.predictor.task = self.task
            sem_seg_head.predictor.step = self.step

            # weighting schedule
            if name.startswith('ade20k_incremental_100-50'):
                self.w = [300, 5, 10, 1000, 0.2, 1]
            elif name.startswith('ade20k_incremental_50-50'):
                self.w = [300, 5, 10, 1000, 0.2, 1]
            elif name.startswith('voc_incremental_15-5'):
                self.w = [300, 5, 10, 1000, 0.2, 1]
            elif name.startswith('voc_incremental_19-1'):
                self.w = [300, 5, 10, 1000, 0.2, 1]
            elif name.startswith('coco_incrementalins'):
                if '40-40' in name:
                    self.w = [120, 2, 10, 1000, 1, 0.4]
                else:
                    self.w = [300, 5, 10, 1000, 0.2, 0.1]
            else:
                self.w = [300, 5, 100, 10000, 1, 0.1]


            # freeze the old queries
            self.tmodel = None
            if self.step > 0:
                sem_seg_head.predictor.query_feat.weight.requires_grad = False
                for s in range(self.step - 1):
                    sem_seg_head.predictor.incre_query_feat_list[s].weight.requires_grad = False

                # get the teacher model
                if not self.cfg.MODEL.MODEL_OLD:
                    # load step 0 teacher model
                    # model_base_cfg = copy.deepcopy(cfg)
                    # model_base_cfg.defrost()
                    # model_base_cfg.MODEL.MODEL_OLD = True
                    # weights = model_base_cfg.MODEL.WEIGHTS
                    # weights = os.path.join('/'.join(weights.split('/')[:-2]),'step0/cur.pth')
                    # model_base_cfg.MODEL.WEIGHTS = weights
                    #
                    # model_base = build_model(model_base_cfg)
                    # model_base.step = 0
                    # model_base.sem_seg_head.predictor.step = 0
                    # DetectionCheckpointer(model_base, save_dir='./cache/debug').resume_or_load(
                    #     model_base_cfg.MODEL.WEIGHTS
                    # )
                    # model_base.eval()

                    # load step t-1 teacher model
                    model_old_cfg = copy.deepcopy(cfg)
                    model_old_cfg.defrost()
                    model_old_cfg.MODEL.MODEL_OLD = True

                    model_old = build_model(model_old_cfg)
                    model_old.step = self.step - 1
                    model_old.sem_seg_head.predictor.step = self.step - 1
                    DetectionCheckpointer(model_old, save_dir='./cache/debug').resume_or_load(
                        model_old_cfg.MODEL.WEIGHTS
                    )
                    model_old.eval()
                    self.tmodel = {'model_base': model_old, 'model_old': model_old}
                else:
                    self.tmodel = None


            self.backbone = backbone
            self.sem_seg_head = sem_seg_head
            self.criterion = criterion
            self.num_queries = num_queries
            self.overlap_threshold = overlap_threshold
            self.object_mask_threshold = object_mask_threshold
            self.metadata = metadata
            if size_divisibility < 0:
                # use backbone size_divisibility if not set
                size_divisibility = self.backbone.size_divisibility
            self.size_divisibility = size_divisibility
            self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
            self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

            # additional args
            self.semantic_on = semantic_on
            self.instance_on = instance_on
            self.panoptic_on = panoptic_on
            self.test_topk_per_image = test_topk_per_image

            if not self.semantic_on:
                assert self.sem_seg_postprocess_before_inference

            self.params = {
                'collapse_channels': 'local',
                'pod_apply': 'all',
                'pod_factor': 1.,
                'prepro': 'pow',
                'spp_scales': [1, 2, 4],
                'pod_options': {"switch": {"after": {"extra_channels": "sum", "factor": 0.00001, "type": "local"}}},
                'use_pod_schedule': True,
            }

            self.num_base_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
            self.num_novel_classes = cfg.MODEL.MASK_FORMER.NUM_INCRE_CLASSES





        @classmethod
        def from_config(cls, cfg):
            backbone = build_backbone(cfg)
            sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

            # Loss parameters:
            deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
            no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

            # loss weights
            class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
            dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
            mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

            # building criterion
            matcher = HungarianMatcher(
                cost_class=class_weight,
                cost_mask=mask_weight,
                cost_dice=dice_weight,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            )

            weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

            if deep_supervision:
                dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
                aux_weight_dict = {}
                for i in range(dec_layers - 1):
                    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
                weight_dict.update(aux_weight_dict)

            losses = ["labels", "masks"]

            num_base_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
            num_novel_classes = cfg.MODEL.MASK_FORMER.NUM_INCRE_CLASSES

            criterion = nn.ModuleDict()

            base_criterion = SetCriterion(
                num_base_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight,
                losses=losses,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
                oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
                importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            )

            novel_criterion = SetCriterion(
                num_novel_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight,
                losses=losses,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
                oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
                importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            )

            criterion['base_criterion'] = base_criterion
            criterion['novel_criterion'] = novel_criterion

            return {
                "cfg": cfg,
                "name": cfg.DATASETS.TRAIN[0],
                "backbone": backbone,
                "sem_seg_head": sem_seg_head,
                "criterion": criterion,
                "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
                "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
                "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
                "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
                "sem_seg_postprocess_before_inference": (
                        cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                        or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                        or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
                ),
                "pixel_mean": cfg.MODEL.PIXEL_MEAN,
                "pixel_std": cfg.MODEL.PIXEL_STD,
                # inference
                "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
                "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
                "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
                "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            }

        @property
        def device(self):
            return self.pixel_mean.device

        def entropy(self,probabilities):
            """Computes the entropy per pixel.

            # References:
                * ESL: Entropy-guided Self-supervised Learning for Domain Adaptation in Semantic Segmentation
                  Saporta et al.
                  CVPR Workshop 2020

            :param probabilities: Tensor of shape (b, c, w, h).
            :return: One entropy per pixel, shape (b, w, h)
            """
            factor = 1 / math.log(probabilities.shape[1] + 1e-8)
            return -factor * torch.mean(probabilities * torch.log(probabilities + 1e-8), dim=1)


        def losskd(self, new_features, old_features, batched_inputs):
            losskd = dict()

            # wpred = [1,0.1]

            kd_groups = old_features['predictions_class'][0].keys()


            for group in kd_groups:
                # format gt
                gt_logits = old_features['predictions_class'][-1][group]
                gt_masks = old_features['predictions_mask'][-1][group]
                pesudo_targets = []
                for mask_cls_result, mask_pred_result in zip(gt_logits, gt_masks):
                    # if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)


                    score_thr = 0.5
                    pesudo_label = {'labels': [], 'masks': [], 'gt_ignore': None}
                    scores = instance_r.scores
                    scores_mask = scores > score_thr
                    # pred_masks = F.interpolate(instance_r.pred_masks[None],(h,w))[0]
                    pred_masks = instance_r.pred_masks[scores_mask]
                    pred_classes = instance_r.pred_classes[scores_mask] + 1
                    pesudo_label['labels'] = pred_classes.to(torch.int64)
                    pesudo_label['masks'] = (pred_masks).to(torch.bool)
                    pesudo_targets.append(pesudo_label)

                # format pred
                outputs = dict()
                pred_logits = [el[group] for el in new_features['predictions_class']]
                pred_masks = [el[group] for el in new_features['predictions_mask']]
                outputs['pred_logits'] = pred_logits[-1]
                outputs['pred_masks'] = pred_masks[-1]
                aux_outputs = []
                for el1, el2 in zip(pred_logits[:-1],pred_masks[:-1]):
                    aux_outputs.append({'pred_logits':el1, 'pred_masks':el2})
                outputs['aux_outputs'] = aux_outputs

                # set loss for a single group queries
                if group == 'step0':
                    criterion = self.criterion['base_criterion']
                    loss_g = criterion(outputs, pesudo_targets)
                else:
                    criterion = self.criterion['novel_criterion']
                    loss_g = criterion(outputs, pesudo_targets)

                for k in list(loss_g.keys()):
                    if k in criterion.weight_dict:
                        loss_g[k] *= criterion.weight_dict[k]
                    else:
                        continue
                losskd[f'loss_{group}'] = sum(loss_g.values())

            # for k in list(losskd.keys()):
            #     if 'step0' in k:
            #         losskd[k] *= wpred[0]
            #     else:
            #         losskd[k] *= wpred[1]


            # kernel = np.ones((10, 10), np.uint8)
            # fg_masks = [(el['sem_seg']!=255).detach().cpu().numpy() for el in batched_inputs]
            # bg_masks = []
            # for mask in fg_masks:
            #     bg_masks.append(mask.astype(np.float))
            #     # bg_masks.append(cv2.dilate(mask.astype(np.float), kernel))
            # bg_masks = torch.from_numpy(1-np.stack(bg_masks,0)[:,None]).to(torch.float)

            wpod = [10,100,0.1]

            # ResNet features KD
            pod_old_features = []
            pod_new_features = []
            for k, v in old_features['res_features'].items():
                _, _, h, w = v.shape
                # masks = F.interpolate(bg_masks,(h,w)).to(v)
                pod_old_features.append(v)

            for k, v in new_features['res_features'].items():
                _, _, h, w = v.shape
                # masks = F.interpolate(bg_masks, (h, w)).to(v)
                pod_new_features.append(v)
            losskd[f'loss_kd_pod_res'] = features_distillation(pod_old_features, pod_new_features, **self.params)*wpod[0]

            # mask features KD
            pod_old_features = []
            pod_new_features = []
            pod_old_features.append(old_features['mask_features'])
            pod_new_features.append(new_features['mask_features'])
            losskd[f'loss_kd_pod_mask'] = features_distillation(pod_old_features, pod_new_features, **self.params)*wpod[1]

            # query KD
            pod_old_features = []
            pod_new_features = []
            kd_query_steps = old_features['predictions_query'][0].keys()
            for k1 in kd_query_steps:
                for old_feature, new_feature in zip(old_features['predictions_query'], new_features['predictions_query']):
                    pod_old_features.append(old_feature[k1].reshape(-1,256)[...,None,None])
                    pod_new_features.append(new_feature[k1].reshape(-1,256)[...,None,None])
            losskd[f'loss_kd_pod_query'] = features_distillation(pod_old_features, pod_new_features, **self.params)*wpod[2]


            return losskd

        def detach_features(self, features):
            if isinstance(features, torch.Tensor):
                return features.detach()
            elif isinstance(features, dict):
                new_dict = dict()
                for k, v in features.items():
                    new_dict[k] = self.detach_features(v)
                return new_dict
            elif isinstance(features, list):
                return [self.detach_features(el) for el in features]
            else:
                raise ValueError("unknow type")

        def unify_features(self, base_features, old_features):
            for old_feat, base_feat in zip(old_features['predictions_class'],base_features['predictions_class']):
                old_feat['step0'] = base_feat['step0']
            for old_feat, base_feat in zip(old_features['predictions_mask'],base_features['predictions_mask']):
                old_feat['step0'] = base_feat['step0']
            for old_feat, base_feat in zip(old_features['predictions_query'],base_features['predictions_query']):
                old_feat['step0'] = base_feat['step0']
            return old_features

        def forward(self, batched_inputs, kd=False):
            if kd:
                return self.forward_old(batched_inputs)
            else:
                return self.forward_new(batched_inputs)

        def forward_old(self, batched_inputs):
            # forward path of the model
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            features = self.backbone(images.tensor)

            outputs = self.sem_seg_head(features)

            # semantic predictions
            sem_logits = self.semantic_inference(outputs['pred_logits'], outputs['pred_masks'])
            outputs['kd_features']['predictions_sem'] = sem_logits

            # return the kd features
            kd_features = outputs['kd_features']
            kd_features['res_features'] = features
            return self.detach_features(kd_features)

        def forward_new(self, batched_inputs):
            # forward path of the model
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            features = self.backbone(images.tensor)

            outputs = self.sem_seg_head(features)

            # semantic predictions
            sem_logits = self.semantic_inference(outputs['pred_logits'], outputs['pred_masks'])
            outputs['kd_features']['predictions_sem'] = sem_logits

            if self.training:
                # mask classification target
                if "instances" in batched_inputs[0]:
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                    gt_ignores = [x["ignore"].to(self.device) for x in batched_inputs]
                    targets = self.prepare_targets(gt_instances, gt_ignores, images)
                else:
                    targets = None

                losses = dict()

                # cal the kd loss
                if self.training and self.tmodel is not None:
                    # base_features = self.tmodel['model_base'](batched_inputs, kd=True)
                    old_features = self.tmodel['model_old'](batched_inputs, kd=True)
                    # old_features = self.unify_features(base_features, old_features)
                    new_features = outputs['kd_features']
                    new_features['res_features'] = features
                    losses.update(self.losskd(new_features, old_features, batched_inputs))

                if 'kd_features' in outputs:
                    outputs.pop('kd_features')

                # bipartite matching-based loss
                if self.step == 0:
                    criterion = self.criterion['base_criterion']
                else:
                    criterion = self.criterion['novel_criterion']
                loss_novel = criterion(outputs, targets)
                # losses.update(criterion(outputs, targets))

                for k in list(loss_novel.keys()):
                    if k in criterion.weight_dict:
                        loss_novel[k] *= criterion.weight_dict[k]
                    else:
                        continue
                loss_novel = sum(loss_novel.values())
                losses[f'loss_step{self.step}'] = loss_novel

                # weighting group loss by number of classes
                weight_sum = self.num_base_classes + self.step * self.num_novel_classes
                for i in range(self.step+1):
                    key = f'loss_step{i}'
                    if i == 0:
                        losses[key] *= self.num_base_classes / weight_sum
                    else:
                        losses[key] *= self.num_novel_classes / weight_sum

                return losses
            else:
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )

                del outputs

                processed_results = []
                for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                        mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    processed_results.append({})

                    if self.sem_seg_postprocess_before_inference:
                        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            mask_pred_result, image_size, height, width
                        )
                        mask_cls_result = mask_cls_result.to(mask_pred_result)

                    # semantic segmentation inference
                    if self.semantic_on:
                        r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                        if not self.sem_seg_postprocess_before_inference:
                            r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                        processed_results[-1]["sem_seg"] = r

                    # panoptic segmentation inference
                    if self.panoptic_on:
                        panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["panoptic_seg"] = panoptic_r

                    # instance segmentation inference
                    if self.instance_on:
                        instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["instances"] = instance_r

                return processed_results

        def prepare_targets(self, targets, gt_ignores, images):
            h_pad, w_pad = images.tensor.shape[-2:]
            new_targets = []
            for targets_per_image, gt_ignore in zip(targets, gt_ignores):
                # pad gt
                gt_masks = targets_per_image.gt_masks
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype,
                                           device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
                new_targets.append(
                    {
                        "labels": targets_per_image.gt_classes,
                        "masks": padded_masks,
                        "gt_ignore": gt_ignore,
                    }
                )
            return new_targets

        def semantic_inference(self, mask_cls, mask_pred):
            mask_cls = F.softmax(mask_cls, dim=-1)[..., 1:]  # the first class is the background
            # mask_cls[...,0] = 0
            # mask_cls = F.softmax(mask_cls, dim=-1)
            mask_pred = mask_pred.sigmoid()
            if len(mask_cls.shape)==2:
                semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            else:
                semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
            return semseg

        def panoptic_inference(self, mask_cls, mask_pred):
            scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
            mask_pred = mask_pred.sigmoid()

            keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[keep]
            cur_mask_cls = mask_cls[keep]
            cur_mask_cls = cur_mask_cls[:, :-1]

            cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

            h, w = cur_masks.shape[-2:]
            panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
            segments_info = []

            current_segment_id = 0

            if cur_masks.shape[0] == 0:
                # We didn't detect any mask :(
                return panoptic_seg, segments_info
            else:
                # take argmax
                cur_mask_ids = cur_prob_masks.argmax(0)
                stuff_memory_list = {}
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()
                    isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                    mask_area = (cur_mask_ids == k).sum().item()
                    original_area = (cur_masks[k] >= 0.5).sum().item()
                    mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        if mask_area / original_area < self.overlap_threshold:
                            continue

                        # merge stuff regions
                        if not isthing:
                            if int(pred_class) in stuff_memory_list.keys():
                                panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                                continue
                            else:
                                stuff_memory_list[int(pred_class)] = current_segment_id + 1

                        current_segment_id += 1
                        panoptic_seg[mask] = current_segment_id

                        segments_info.append(
                            {
                                "id": current_segment_id,
                                "isthing": bool(isthing),
                                "category_id": int(pred_class),
                            }
                        )

                return panoptic_seg, segments_info

        def instance_inference(self, mask_cls, mask_pred):
            # mask_pred is already processed to have the same shape as original input
            image_size = mask_pred.shape[-2:]

            # [Q, K]
            # scores = F.softmax(mask_cls, dim=-1)[:, :-1]
            scores = F.softmax(mask_cls, dim=-1)[:, 1:]
            num_queries, num_classes = scores.shape
            labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
            # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
            test_topk_per_image = min(len(scores),self.test_topk_per_image)
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(test_topk_per_image, sorted=False)
            labels_per_image = labels[topk_indices]

            topk_indices = topk_indices // num_classes
            # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
            mask_pred = mask_pred[topk_indices]

            # if this is panoptic segmentation, we only keep the "thing" classes
            if self.panoptic_on:
                keep = torch.zeros_like(scores_per_image).bool()
                for i, lab in enumerate(labels_per_image):
                    keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]
                mask_pred = mask_pred[keep]

            result = Instances(image_size)
            # mask (before sigmoid)
            result.pred_masks = (mask_pred > 0).float()
            result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
            # Uncomment the following to get boxes from masks (this is slow)
            # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

            # calculate average mask prob
            mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
                    result.pred_masks.flatten(1).sum(1) + 1e-6)
            result.scores = scores_per_image * mask_scores_per_image
            result.pred_classes = labels_per_image

            thr = 0.5
            pos = result._fields['scores'] > thr
            result._fields['pred_masks'] = result._fields['pred_masks'][pos]
            result._fields['pred_boxes'] = result._fields['pred_boxes'][pos]
            result._fields['pred_classes'] = result._fields['pred_classes'][pos]
            result._fields['scores'] = result._fields['scores'][pos]
            return result

elif version == 'cisdq_ade':
    # query freeze : True
    # query kd : True
    # prediction kd : True
    # query group attention mask : True



    @META_ARCH_REGISTRY.register()
    class IncrementalMaskFormer(nn.Module):
        """
        Main class for mask classification semantic segmentation architectures.
        """


        @configurable
        def __init__(
                self,
                *,
                cfg: None,
                name: str,
                backbone: Backbone,
                sem_seg_head: nn.Module,
                criterion: nn.Module,
                num_queries: int,
                object_mask_threshold: float,
                overlap_threshold: float,
                metadata,
                size_divisibility: int,
                sem_seg_postprocess_before_inference: bool,
                pixel_mean: Tuple[float],
                pixel_std: Tuple[float],
                # inference
                semantic_on: bool,
                panoptic_on: bool,
                instance_on: bool,
                test_topk_per_image: int,
        ):
            """
            Args:
                backbone: a backbone module, must follow detectron2's backbone interface
                sem_seg_head: a module that predicts semantic segmentation from backbone features
                criterion: a module that defines the loss
                num_queries: int, number of queries
                object_mask_threshold: float, threshold to filter query based on classification score
                    for panoptic segmentation inference
                overlap_threshold: overlap threshold used in general inference for panoptic segmentation
                metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                    segmentation inference
                size_divisibility: Some backbones require the input height and width to be divisible by a
                    specific integer. We can use this to override such requirement.
                sem_seg_postprocess_before_inference: whether to resize the prediction back
                    to original input size before semantic segmentation inference or after.
                    For high-resolution dataset like Mapillary, resizing predictions before
                    inference will cause OOM error.
                pixel_mean, pixel_std: list or tuple with #channels element, representing
                    the per-channel mean and std to be used to normalize the input image
                semantic_on: bool, whether to output semantic segmentation prediction
                instance_on: bool, whether to output instance segmentation prediction
                panoptic_on: bool, whether to output panoptic segmentation prediction
                test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
            """
            super().__init__()
            self.cfg = copy.deepcopy(cfg)
            dataset, _, task, step = name.split('_')
            self.dataset = dataset
            self.task = task
            self.step = int(step)

            sem_seg_head.predictor.task = self.task
            sem_seg_head.predictor.step = self.step

            # weighting schedule
            if name.startswith('ade20k_incremental_100-50'):
                self.w = [300, 5, 10, 1000, 0.2, 1]
            elif name.startswith('ade20k_incremental_50-50'):
                self.w = [300, 5, 10, 1000, 0.2, 1]
            elif name.startswith('voc_incremental_15-5'):
                self.w = [300, 5, 10, 1000, 0.2, 1]
            elif name.startswith('voc_incremental_19-1'):
                self.w = [300, 5, 10, 1000, 0.2, 1]
            elif name.startswith('coco_incrementalins'):
                if '40-40' in name:
                    self.w = [120, 2, 10, 1000, 1, 0.4]
                else:
                    self.w = [300, 5, 10, 1000, 0.2, 0.1]
            else:
                self.w = [300, 5, 100, 10000, 1, 0.1]

            # freeze the old queries
            self.tmodel = None
            if self.step > 0:
                sem_seg_head.predictor.query_feat.weight.requires_grad = False
                for s in range(self.step - 1):
                    sem_seg_head.predictor.incre_query_feat_list[s].weight.requires_grad = False

                # get the teacher model
                if not self.cfg.MODEL.MODEL_OLD:
                    # load step 0 teacher model
                    model_base_cfg = copy.deepcopy(cfg)
                    model_base_cfg.defrost()
                    model_base_cfg.MODEL.MODEL_OLD = True
                    weights = model_base_cfg.MODEL.WEIGHTS
                    weights = os.path.join('/'.join(weights.split('/')[:-2]),'step0/cur.pth')
                    model_base_cfg.MODEL.WEIGHTS = weights

                    model_base = build_model(model_base_cfg)
                    model_base.step = 0
                    model_base.sem_seg_head.predictor.step = 0
                    DetectionCheckpointer(model_base, save_dir='./cache/debug').resume_or_load(
                        model_base_cfg.MODEL.WEIGHTS
                    )
                    model_base.eval()

                    # load step t-1 teacher model
                    model_old_cfg = copy.deepcopy(cfg)
                    model_old_cfg.defrost()
                    model_old_cfg.MODEL.MODEL_OLD = True

                    model_old = build_model(model_old_cfg)
                    model_old.step = self.step - 1
                    model_old.sem_seg_head.predictor.step = self.step - 1
                    DetectionCheckpointer(model_old, save_dir='./cache/debug').resume_or_load(
                        model_old_cfg.MODEL.WEIGHTS
                    )
                    model_old.eval()
                    self.tmodel = {'model_base': model_base, 'model_old': model_old}
                else:
                    self.tmodel = None


            self.backbone = backbone
            self.sem_seg_head = sem_seg_head
            self.criterion = criterion
            self.num_queries = num_queries
            self.overlap_threshold = overlap_threshold
            self.object_mask_threshold = object_mask_threshold
            self.metadata = metadata
            if size_divisibility < 0:
                # use backbone size_divisibility if not set
                size_divisibility = self.backbone.size_divisibility
            self.size_divisibility = size_divisibility
            self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
            self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

            # additional args
            self.semantic_on = semantic_on
            self.instance_on = instance_on
            self.panoptic_on = panoptic_on
            self.test_topk_per_image = test_topk_per_image

            if not self.semantic_on:
                assert self.sem_seg_postprocess_before_inference

            self.l1_loss = nn.SmoothL1Loss()

            self.params = {
                'collapse_channels': 'local',
                'pod_apply': 'all',
                'pod_factor': 1.,
                'prepro': 'pow',
                'spp_scales': [1, 2, 4],
                'pod_options': {"switch": {"after": {"extra_channels": "sum", "factor": 0.00001, "type": "local"}}},
                'use_pod_schedule': True,
            }





        @classmethod
        def from_config(cls, cfg):
            backbone = build_backbone(cfg)
            sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

            # Loss parameters:
            deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
            no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

            # loss weights
            class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
            dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
            mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

            # building criterion
            matcher = HungarianMatcher(
                cost_class=class_weight,
                cost_mask=mask_weight,
                cost_dice=dice_weight,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            )

            weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

            if deep_supervision:
                dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
                aux_weight_dict = {}
                for i in range(dec_layers - 1):
                    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
                weight_dict.update(aux_weight_dict)

            losses = ["labels", "masks"]

            dataset, seg, task, step = cfg.DATASETS.TRAIN[0].split('_')

            if dataset == 'ade20k':
                if seg == 'incrementalins':
                    from mask2former.data.datasets.register_ade20k_incremental_instance import tasks_ade_ins
                    training_classes = tasks_ade_ins[task][int(step)]
                else:
                    from mask2former.data.datasets.register_ade20k_incremental import tasks_ade
                    training_classes = tasks_ade[task][int(step)]
                cr_num_classes = len(training_classes)
            elif dataset == 'voc':
                from mask2former.data.datasets.register_voc_incremental import tasks_voc
                training_classes = tasks_voc[task][int(step)]
                cr_num_classes = len(training_classes)
            elif dataset == 'coco':
                from mask2former.data.datasets.register_coco_incremental_instance import tasks_coco_ins
                training_classes = tasks_coco_ins[task][int(step)]
                cr_num_classes = len(training_classes)
            else:
                print(f'no dataset {dataset}')
                raise ValueError('No dataset')

            criterion = SetCriterion(
                cr_num_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight,
                losses=losses,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
                oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
                importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            )

            return {
                "cfg": cfg,
                "name": cfg.DATASETS.TRAIN[0],
                "backbone": backbone,
                "sem_seg_head": sem_seg_head,
                "criterion": criterion,
                "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
                "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
                "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
                "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
                "sem_seg_postprocess_before_inference": (
                        cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                        or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                        or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
                ),
                "pixel_mean": cfg.MODEL.PIXEL_MEAN,
                "pixel_std": cfg.MODEL.PIXEL_STD,
                # inference
                "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
                "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
                "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
                "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            }

        @property
        def device(self):
            return self.pixel_mean.device

        def losskd(self, new_features, old_features):
            losskd = dict()

            # wpred = [300,150]
            wpred = self.w[0:2]

            # Prediction KD
            predictions_classs = old_features['predictions_class']
            ppredictions_classs = new_features['predictions_class']
            predictions_masks = old_features['predictions_mask']
            ppredictions_masks = new_features['predictions_mask']
            nl = [9]
            nk = predictions_classs[0].keys()
            for l in nl:
                total_nq = sum([ppredictions_classs[l][k].shape[1] for k in nk])
                losskd[f'loss_kd_class{l}'] = 0
                losskd[f'loss_kd_mask{l}'] = 0
                for k in nk:
                    pred_class = ppredictions_classs[l][k]
                    gt_class = predictions_classs[l][k]

                    b, nq, c = pred_class.shape

                    pred = pred_class.reshape(-1, c).softmax(-1)
                    gt = gt_class.reshape(-1, c).softmax(-1)
                    # 1000 for 11 task, 300 for 2 or 3 task
                    step_loss_kd_class = (F.kl_div(pred.log(), gt, reduction='sum') / (b * nq) * wpred[0]) * nq / total_nq
                    losskd[f'loss_kd_class{l}'] += step_loss_kd_class

                    pred_mask = ppredictions_masks[l][k]
                    gt_mask = predictions_masks[l][k]

                    step_loss_kd_mask = self.l1_loss(pred_mask, gt_mask) * wpred[1] * nq / total_nq
                    losskd[f'loss_kd_mask{l}'] += step_loss_kd_mask

                    # b, nq, h, w = pred_mask.shape
                    # pred = pred_mask.reshape(-1, h, w)
                    # gt = (gt_mask.reshape(-1, h, w) > 0).to(torch.int64)
                    # step_loss_kd_mask = (self.dice_loss(pred, gt) * wpred[1]) * nq / total_nq
                    # losskd[f'loss_kd_mask{l}'] += step_loss_kd_mask

            wpod = self.w[2:5]

            # ResNet features KD
            pod_old_features = []
            pod_new_features = []
            for k, v in old_features['res_features'].items():
                pod_old_features.append(v)

            for k, v in new_features['res_features'].items():
                pod_new_features.append(v)
            losskd[f'loss_kd_pod_res'] = features_distillation(pod_old_features, pod_new_features, **self.params)*wpod[0]

            # mask features KD
            pod_old_features = []
            pod_new_features = []
            pod_old_features.append(old_features['mask_features'])
            pod_new_features.append(new_features['mask_features'])
            losskd[f'loss_kd_pod_mask'] = features_distillation(pod_old_features, pod_new_features, **self.params)*wpod[1]

            # query KD
            pod_old_features = []
            pod_new_features = []
            kd_query_steps = old_features['predictions_query'][0].keys()
            for k1 in kd_query_steps:
                for old_feature, new_feature in zip(old_features['predictions_query'], new_features['predictions_query']):
                    pod_old_features.append(old_feature[k1].reshape(-1,256)[...,None,None])
                    pod_new_features.append(new_feature[k1].reshape(-1,256)[...,None,None])
            losskd[f'loss_kd_pod_query'] = features_distillation(pod_old_features, pod_new_features, **self.params)*wpod[2]


            return losskd

        def detach_features(self, features):
            if isinstance(features, torch.Tensor):
                return features.detach()
            elif isinstance(features, dict):
                new_dict = dict()
                for k, v in features.items():
                    new_dict[k] = self.detach_features(v)
                return new_dict
            elif isinstance(features, list):
                return [self.detach_features(el) for el in features]
            else:
                raise ValueError("unknow type")

        def unify_features(self, base_features, old_features):
            for old_feat, base_feat in zip(old_features['predictions_class'],base_features['predictions_class']):
                old_feat['step0'] = base_feat['step0']
            for old_feat, base_feat in zip(old_features['predictions_mask'],base_features['predictions_mask']):
                old_feat['step0'] = base_feat['step0']
            for old_feat, base_feat in zip(old_features['predictions_query'],base_features['predictions_query']):
                old_feat['step0'] = base_feat['step0']
            return old_features

        def forward(self, batched_inputs, kd=False):
            if kd:
                return self.forward_old(batched_inputs)
            else:
                return self.forward_new(batched_inputs)

        def forward_old(self, batched_inputs):
            # forward path of the model
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            features = self.backbone(images.tensor)

            outputs = self.sem_seg_head(features)

            # semantic predictions
            sem_logits = self.semantic_inference(outputs['pred_logits'], outputs['pred_masks'])
            outputs['kd_features']['predictions_sem'] = sem_logits

            # return the kd features
            kd_features = outputs['kd_features']
            kd_features['res_features'] = features
            return self.detach_features(kd_features)

        def forward_new(self, batched_inputs):
            # forward path of the model
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            features = self.backbone(images.tensor)

            outputs = self.sem_seg_head(features)

            # semantic predictions
            sem_logits = self.semantic_inference(outputs['pred_logits'], outputs['pred_masks'])
            outputs['kd_features']['predictions_sem'] = sem_logits

            if self.training:
                # mask classification target
                if "instances" in batched_inputs[0]:
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                    gt_ignores = [x["ignore"].to(self.device) for x in batched_inputs]
                    targets = self.prepare_targets(gt_instances, gt_ignores, images)
                else:
                    targets = None

                losses = dict()

                # cal the kd loss
                if self.training and self.tmodel is not None:
                    base_features = self.tmodel['model_base'](batched_inputs, kd=True)
                    old_features = self.tmodel['model_old'](batched_inputs, kd=True)
                    old_features = self.unify_features(base_features, old_features)
                    new_features = outputs['kd_features']
                    new_features['res_features'] = features
                    losses.update(self.losskd(new_features, old_features))

                if 'kd_features' in outputs:
                    outputs.pop('kd_features')


                loss_novel = self.criterion(outputs, targets)

                for k in list(loss_novel.keys()):
                    if k in self.criterion.weight_dict:
                        loss_novel[k] *= self.criterion.weight_dict[k]
                    else:
                        continue
                loss_novel = sum(loss_novel.values())
                if self.step == 0:
                    wnovel = 1.
                else:
                    wnovel = self.w[5]
                losses[f'loss_step{self.step}'] = loss_novel * wnovel

                return losses
            else:
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )

                del outputs

                processed_results = []
                for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                        mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    processed_results.append({})

                    if self.sem_seg_postprocess_before_inference:
                        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            mask_pred_result, image_size, height, width
                        )
                        mask_cls_result = mask_cls_result.to(mask_pred_result)

                    # semantic segmentation inference
                    if self.semantic_on:
                        r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                        if not self.sem_seg_postprocess_before_inference:
                            r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                        processed_results[-1]["sem_seg"] = r

                    # panoptic segmentation inference
                    if self.panoptic_on:
                        panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["panoptic_seg"] = panoptic_r

                    # instance segmentation inference
                    if self.instance_on:
                        instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["instances"] = instance_r

                return processed_results

        def prepare_targets(self, targets, gt_ignores, images):
            h_pad, w_pad = images.tensor.shape[-2:]
            new_targets = []
            for targets_per_image, gt_ignore in zip(targets, gt_ignores):
                # pad gt
                gt_masks = targets_per_image.gt_masks
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype,
                                           device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
                new_targets.append(
                    {
                        "labels": targets_per_image.gt_classes,
                        "masks": padded_masks,
                        "gt_ignore": gt_ignore,
                    }
                )
            return new_targets

        def semantic_inference(self, mask_cls, mask_pred):

            mask_cls = F.softmax(mask_cls, dim=-1)[..., 1:]  # the first class is the background
            mask_pred = mask_pred.sigmoid()
            if len(mask_cls.shape)==2:
                semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            else:
                semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
            return semseg

        def panoptic_inference(self, mask_cls, mask_pred):
            scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
            mask_pred = mask_pred.sigmoid()

            keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[keep]
            cur_mask_cls = mask_cls[keep]
            cur_mask_cls = cur_mask_cls[:, :-1]

            cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

            h, w = cur_masks.shape[-2:]
            panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
            segments_info = []

            current_segment_id = 0

            if cur_masks.shape[0] == 0:
                # We didn't detect any mask :(
                return panoptic_seg, segments_info
            else:
                # take argmax
                cur_mask_ids = cur_prob_masks.argmax(0)
                stuff_memory_list = {}
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()
                    isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                    mask_area = (cur_mask_ids == k).sum().item()
                    original_area = (cur_masks[k] >= 0.5).sum().item()
                    mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        if mask_area / original_area < self.overlap_threshold:
                            continue

                        # merge stuff regions
                        if not isthing:
                            if int(pred_class) in stuff_memory_list.keys():
                                panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                                continue
                            else:
                                stuff_memory_list[int(pred_class)] = current_segment_id + 1

                        current_segment_id += 1
                        panoptic_seg[mask] = current_segment_id

                        segments_info.append(
                            {
                                "id": current_segment_id,
                                "isthing": bool(isthing),
                                "category_id": int(pred_class),
                            }
                        )

                return panoptic_seg, segments_info

        def instance_inference(self, mask_cls, mask_pred):
            # mask_pred is already processed to have the same shape as original input
            image_size = mask_pred.shape[-2:]

            # [Q, K]
            # scores = F.softmax(mask_cls, dim=-1)[:, :-1]
            scores = F.softmax(mask_cls, dim=-1)[:, 1:]
            num_queries, num_classes = scores.shape
            labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
            # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
            labels_per_image = labels[topk_indices]

            topk_indices = topk_indices // num_classes
            # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
            mask_pred = mask_pred[topk_indices]

            # if this is panoptic segmentation, we only keep the "thing" classes
            if self.panoptic_on:
                keep = torch.zeros_like(scores_per_image).bool()
                for i, lab in enumerate(labels_per_image):
                    keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]
                mask_pred = mask_pred[keep]

            result = Instances(image_size)
            # mask (before sigmoid)
            result.pred_masks = (mask_pred > 0).float()
            result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
            # Uncomment the following to get boxes from masks (this is slow)
            # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

            # calculate average mask prob
            mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
                    result.pred_masks.flatten(1).sum(1) + 1e-6)
            result.scores = scores_per_image * mask_scores_per_image
            result.pred_classes = labels_per_image
            return result

elif version == 'cisdq_voc':
    # query freeze : True
    # query kd : True
    # prediction kd : True
    # query group attention mask : True



    @META_ARCH_REGISTRY.register()
    class IncrementalMaskFormer(nn.Module):
        """
        Main class for mask classification semantic segmentation architectures.
        """


        @configurable
        def __init__(
                self,
                *,
                cfg: None,
                name: str,
                backbone: Backbone,
                sem_seg_head: nn.Module,
                criterion: nn.Module,
                num_queries: int,
                object_mask_threshold: float,
                overlap_threshold: float,
                metadata,
                size_divisibility: int,
                sem_seg_postprocess_before_inference: bool,
                pixel_mean: Tuple[float],
                pixel_std: Tuple[float],
                # inference
                semantic_on: bool,
                panoptic_on: bool,
                instance_on: bool,
                test_topk_per_image: int,
        ):
            """
            Args:
                backbone: a backbone module, must follow detectron2's backbone interface
                sem_seg_head: a module that predicts semantic segmentation from backbone features
                criterion: a module that defines the loss
                num_queries: int, number of queries
                object_mask_threshold: float, threshold to filter query based on classification score
                    for panoptic segmentation inference
                overlap_threshold: overlap threshold used in general inference for panoptic segmentation
                metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                    segmentation inference
                size_divisibility: Some backbones require the input height and width to be divisible by a
                    specific integer. We can use this to override such requirement.
                sem_seg_postprocess_before_inference: whether to resize the prediction back
                    to original input size before semantic segmentation inference or after.
                    For high-resolution dataset like Mapillary, resizing predictions before
                    inference will cause OOM error.
                pixel_mean, pixel_std: list or tuple with #channels element, representing
                    the per-channel mean and std to be used to normalize the input image
                semantic_on: bool, whether to output semantic segmentation prediction
                instance_on: bool, whether to output instance segmentation prediction
                panoptic_on: bool, whether to output panoptic segmentation prediction
                test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
            """
            super().__init__()
            self.cfg = copy.deepcopy(cfg)
            dataset, _, task, step = name.split('_')
            self.dataset = dataset
            self.task = task
            self.step = int(step)

            sem_seg_head.predictor.task = self.task
            sem_seg_head.predictor.step = self.step

            # weighting schedule
            if name.startswith('ade20k_incremental_100-50'):
                self.w = [300, 5, 10, 1000, 0.2, 1]
            elif name.startswith('ade20k_incremental_50-50'):
                self.w = [300, 5, 10, 1000, 0.2, 1]
            elif name.startswith('voc_incremental_15-5'):
                self.w = [300, 5, 100, 10000, 1, 0.1]
            elif name.startswith('voc_incremental_19-1'):
                self.w = [300, 5, 1, 100, 0.1, 1]
            elif name.startswith('voc_incremental_15-1'):
                self.w = [300, 5, 100, 10000, 1, 0.1]
            elif name.startswith('voc_incremental_10-1'):
                self.w = [300, 5, 100, 10000, 1, 0.1]
            elif name.startswith('coco_incrementalins'):
                if '40-40' in name:
                    self.w = [120, 2, 10, 1000, 1, 0.4]
                else:
                    self.w = [300, 5, 10, 1000, 0.2, 0.1]
            else:
                self.w = [300, 5, 100, 10000, 1, 0.1]

            # freeze the old queries
            self.tmodel = None
            if self.step > 0:
                sem_seg_head.predictor.query_feat.weight.requires_grad = False
                for s in range(self.step - 1):
                    sem_seg_head.predictor.incre_query_feat_list[s].weight.requires_grad = False

                # get the teacher model
                if not self.cfg.MODEL.MODEL_OLD:

                    # load step t-1 teacher model
                    model_old_cfg = copy.deepcopy(cfg)
                    model_old_cfg.defrost()
                    model_old_cfg.MODEL.MODEL_OLD = True

                    model_old = build_model(model_old_cfg)
                    model_old.step = self.step - 1
                    model_old.sem_seg_head.predictor.step = self.step - 1
                    DetectionCheckpointer(model_old, save_dir='./cache/debug').resume_or_load(
                        model_old_cfg.MODEL.WEIGHTS
                    )
                    model_old.eval()
                    self.tmodel = {'model_old': model_old}
                else:
                    self.tmodel = None


            self.backbone = backbone
            self.sem_seg_head = sem_seg_head
            self.criterion = criterion
            self.num_queries = num_queries
            self.overlap_threshold = overlap_threshold
            self.object_mask_threshold = object_mask_threshold
            self.metadata = metadata
            if size_divisibility < 0:
                # use backbone size_divisibility if not set
                size_divisibility = self.backbone.size_divisibility
            self.size_divisibility = size_divisibility
            self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
            self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

            # additional args
            self.semantic_on = semantic_on
            self.instance_on = instance_on
            self.panoptic_on = panoptic_on
            self.test_topk_per_image = test_topk_per_image

            if not self.semantic_on:
                assert self.sem_seg_postprocess_before_inference

            self.l1_loss = nn.SmoothL1Loss()

            self.params = {
                'collapse_channels': 'local',
                'pod_apply': 'all',
                'pod_factor': 1.,
                'prepro': 'pow',
                'spp_scales': [1, 2, 4],
                'pod_options': {"switch": {"after": {"extra_channels": "sum", "factor": 0.00001, "type": "local"}}},
                'use_pod_schedule': True,
            }





        @classmethod
        def from_config(cls, cfg):
            backbone = build_backbone(cfg)
            sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

            # Loss parameters:
            deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
            no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

            # loss weights
            class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
            dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
            mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

            # building criterion
            matcher = HungarianMatcher(
                cost_class=class_weight,
                cost_mask=mask_weight,
                cost_dice=dice_weight,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            )

            weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

            if deep_supervision:
                dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
                aux_weight_dict = {}
                for i in range(dec_layers - 1):
                    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
                weight_dict.update(aux_weight_dict)

            losses = ["labels", "masks"]

            dataset, seg, task, step = cfg.DATASETS.TRAIN[0].split('_')

            if dataset == 'ade20k':
                if seg == 'incrementalins':
                    from mask2former.data.datasets.register_ade20k_incremental_instance import tasks_ade_ins
                    training_classes = tasks_ade_ins[task][int(step)]
                else:
                    from mask2former.data.datasets.register_ade20k_incremental import tasks_ade
                    training_classes = tasks_ade[task][int(step)]
                cr_num_classes = len(training_classes)
            elif dataset == 'voc':
                from mask2former.data.datasets.register_voc_incremental import tasks_voc
                training_classes = tasks_voc[task][int(step)]
                cr_num_classes = len(training_classes)
            elif dataset == 'coco':
                from mask2former.data.datasets.register_coco_incremental_instance import tasks_coco_ins
                training_classes = tasks_coco_ins[task][int(step)]
                cr_num_classes = len(training_classes)
            else:
                print(f'no dataset {dataset}')
                raise ValueError('No dataset')

            criterion = SetCriterion(
                cr_num_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight,
                losses=losses,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
                oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
                importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            )

            return {
                "cfg": cfg,
                "name": cfg.DATASETS.TRAIN[0],
                "backbone": backbone,
                "sem_seg_head": sem_seg_head,
                "criterion": criterion,
                "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
                "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
                "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
                "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
                "sem_seg_postprocess_before_inference": (
                        cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                        or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                        or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
                ),
                "pixel_mean": cfg.MODEL.PIXEL_MEAN,
                "pixel_std": cfg.MODEL.PIXEL_STD,
                # inference
                "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
                "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
                "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
                "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            }

        @property
        def device(self):
            return self.pixel_mean.device

        def losskd(self, new_features, old_features, fg_segs):
            losskd = dict()

            # wpred = [300,150]
            wpred = self.w[0:2]

            fg_segs = fg_segs[:,None]
            fg_segs = F.interpolate(fg_segs.to(torch.float), old_features['predictions_mask'][-1]['step0'].shape[-2:])
            fg_segs = fg_segs.to(torch.bool)

            # Prediction KD
            predictions_classs = old_features['predictions_class']
            ppredictions_classs = new_features['predictions_class']
            predictions_masks = old_features['predictions_mask']
            ppredictions_masks = new_features['predictions_mask']
            nl = [9]
            nk = predictions_classs[0].keys()
            for l in nl:
                total_nq = sum([ppredictions_classs[l][k].shape[1] for k in nk])
                losskd[f'loss_kd_class{l}'] = 0
                losskd[f'loss_kd_mask{l}'] = 0
                for k in nk:
                    pred_class = ppredictions_classs[l][k]
                    gt_class = predictions_classs[l][k]

                    b, nq, c = pred_class.shape

                    pred = pred_class.reshape(-1, c).softmax(-1)
                    gt = gt_class.reshape(-1, c).softmax(-1)
                    # 1000 for 11 task, 300 for 2 or 3 task
                    step_loss_kd_class = (F.kl_div(pred.log(), gt, reduction='sum') / (b * nq) * wpred[0]) * nq / total_nq
                    losskd[f'loss_kd_class{l}'] += step_loss_kd_class

                    pred_mask = ppredictions_masks[l][k]
                    gt_mask = predictions_masks[l][k]

                    # mask the background
                    # gt_mask[fg_segs.expand_as(gt_mask)] = 0

                    step_loss_kd_mask = self.l1_loss(pred_mask, gt_mask) * wpred[1] * nq / total_nq
                    losskd[f'loss_kd_mask{l}'] += step_loss_kd_mask

                    # b, nq, h, w = pred_mask.shape
                    # pred = pred_mask.reshape(-1, h, w)
                    # gt = (gt_mask.reshape(-1, h, w) > 0).to(torch.int64)
                    # step_loss_kd_mask = (self.dice_loss(pred, gt) * wpred[1]) * nq / total_nq
                    # losskd[f'loss_kd_mask{l}'] += step_loss_kd_mask

            wpod = self.w[2:5]

            # ResNet features KD
            pod_old_features = []
            pod_new_features = []
            for k, v in old_features['res_features'].items():
                pod_old_features.append(v)

            for k, v in new_features['res_features'].items():
                pod_new_features.append(v)
            losskd[f'loss_kd_pod_res'] = features_distillation(pod_old_features, pod_new_features, **self.params)*wpod[0]

            # mask features KD
            pod_old_features = []
            pod_new_features = []
            pod_old_features.append(old_features['mask_features'])
            pod_new_features.append(new_features['mask_features'])
            losskd[f'loss_kd_pod_mask'] = features_distillation(pod_old_features, pod_new_features, **self.params)*wpod[1]

            # query KD
            pod_old_features = []
            pod_new_features = []
            kd_query_steps = old_features['predictions_query'][0].keys()
            for k1 in kd_query_steps:
                for old_feature, new_feature in zip(old_features['predictions_query'], new_features['predictions_query']):
                    pod_old_features.append(old_feature[k1].reshape(-1,256)[...,None,None])
                    pod_new_features.append(new_feature[k1].reshape(-1,256)[...,None,None])
            losskd[f'loss_kd_pod_query'] = features_distillation(pod_old_features, pod_new_features, **self.params)*wpod[2]


            return losskd

        def detach_features(self, features):
            if isinstance(features, torch.Tensor):
                return features.detach()
            elif isinstance(features, dict):
                new_dict = dict()
                for k, v in features.items():
                    new_dict[k] = self.detach_features(v)
                return new_dict
            elif isinstance(features, list):
                return [self.detach_features(el) for el in features]
            else:
                raise ValueError("unknow type")

        def unify_features(self, base_features, old_features):
            for old_feat, base_feat in zip(old_features['predictions_class'],base_features['predictions_class']):
                old_feat['step0'] = base_feat['step0']
            for old_feat, base_feat in zip(old_features['predictions_mask'],base_features['predictions_mask']):
                old_feat['step0'] = base_feat['step0']
            for old_feat, base_feat in zip(old_features['predictions_query'],base_features['predictions_query']):
                old_feat['step0'] = base_feat['step0']
            return old_features

        def forward(self, batched_inputs, kd=False):
            if kd:
                return self.forward_old(batched_inputs)
            else:
                return self.forward_new(batched_inputs)

        def forward_old(self, batched_inputs):
            # forward path of the model
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            features = self.backbone(images.tensor)

            outputs = self.sem_seg_head(features)

            # semantic predictions
            sem_logits = self.semantic_inference(outputs['pred_logits'], outputs['pred_masks'])
            outputs['kd_features']['predictions_sem'] = sem_logits

            # return the kd features
            kd_features = outputs['kd_features']
            kd_features['res_features'] = features
            return self.detach_features(kd_features)

        def forward_new(self, batched_inputs):
            # forward path of the model
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            features = self.backbone(images.tensor)

            outputs = self.sem_seg_head(features)

            # semantic predictions
            sem_logits = self.semantic_inference(outputs['pred_logits'], outputs['pred_masks'])
            outputs['kd_features']['predictions_sem'] = sem_logits

            if self.training:
                # mask classification target
                if "instances" in batched_inputs[0]:
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                    gt_ignores = [x["ignore"].to(self.device) for x in batched_inputs]
                    targets = self.prepare_targets(gt_instances, gt_ignores, images)
                else:
                    targets = None

                losses = dict()

                # cal the kd loss
                if self.training and self.tmodel is not None:
                    old_features = self.tmodel['model_old'](batched_inputs, kd=True)
                    new_features = outputs['kd_features']
                    new_features['res_features'] = features
                    fg_segs = torch.stack([el['fg_seg'] for el in batched_inputs],0)
                    losses.update(self.losskd(new_features, old_features, fg_segs))

                if 'kd_features' in outputs:
                    outputs.pop('kd_features')


                loss_novel = self.criterion(outputs, targets)

                for k in list(loss_novel.keys()):
                    if k in self.criterion.weight_dict:
                        loss_novel[k] *= self.criterion.weight_dict[k]
                    else:
                        continue
                loss_novel = sum(loss_novel.values())
                if self.step == 0:
                    wnovel = 1.
                else:
                    wnovel = self.w[5]
                losses[f'loss_step{self.step}'] = loss_novel * wnovel

                return losses
            else:
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )

                del outputs

                processed_results = []
                for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                        mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    processed_results.append({})

                    if self.sem_seg_postprocess_before_inference:
                        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            mask_pred_result, image_size, height, width
                        )
                        mask_cls_result = mask_cls_result.to(mask_pred_result)

                    # semantic segmentation inference
                    if self.semantic_on:
                        r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                        if not self.sem_seg_postprocess_before_inference:
                            r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                        processed_results[-1]["sem_seg"] = r

                    # panoptic segmentation inference
                    if self.panoptic_on:
                        panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["panoptic_seg"] = panoptic_r

                    # instance segmentation inference
                    if self.instance_on:
                        instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["instances"] = instance_r

                return processed_results

        def prepare_targets(self, targets, gt_ignores, images):
            h_pad, w_pad = images.tensor.shape[-2:]
            new_targets = []
            for targets_per_image, gt_ignore in zip(targets, gt_ignores):
                # pad gt
                gt_masks = targets_per_image.gt_masks
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype,
                                           device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
                new_targets.append(
                    {
                        "labels": targets_per_image.gt_classes,
                        "masks": padded_masks,
                        "gt_ignore": gt_ignore,
                    }
                )
            return new_targets

        def semantic_inference(self, mask_cls, mask_pred):
            
            mask_cls = F.softmax(mask_cls, dim=-1)[..., 1:]  # the first class is the background
            mask_pred = mask_pred.sigmoid()
            threshold = 0.3
            if len(mask_cls.shape)==2:
                semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
                semseg[0] = threshold
            else:
                semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
                semseg[:, 0] = threshold
            return semseg

        def panoptic_inference(self, mask_cls, mask_pred):
            scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
            mask_pred = mask_pred.sigmoid()

            keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[keep]
            cur_mask_cls = mask_cls[keep]
            cur_mask_cls = cur_mask_cls[:, :-1]

            cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

            h, w = cur_masks.shape[-2:]
            panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
            segments_info = []

            current_segment_id = 0

            if cur_masks.shape[0] == 0:
                # We didn't detect any mask :(
                return panoptic_seg, segments_info
            else:
                # take argmax
                cur_mask_ids = cur_prob_masks.argmax(0)
                stuff_memory_list = {}
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()
                    isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                    mask_area = (cur_mask_ids == k).sum().item()
                    original_area = (cur_masks[k] >= 0.5).sum().item()
                    mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        if mask_area / original_area < self.overlap_threshold:
                            continue

                        # merge stuff regions
                        if not isthing:
                            if int(pred_class) in stuff_memory_list.keys():
                                panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                                continue
                            else:
                                stuff_memory_list[int(pred_class)] = current_segment_id + 1

                        current_segment_id += 1
                        panoptic_seg[mask] = current_segment_id

                        segments_info.append(
                            {
                                "id": current_segment_id,
                                "isthing": bool(isthing),
                                "category_id": int(pred_class),
                            }
                        )

                return panoptic_seg, segments_info

        def instance_inference(self, mask_cls, mask_pred):
            # mask_pred is already processed to have the same shape as original input
            image_size = mask_pred.shape[-2:]

            # [Q, K]
            # scores = F.softmax(mask_cls, dim=-1)[:, :-1]
            scores = F.softmax(mask_cls, dim=-1)[:, 1:]
            num_queries, num_classes = scores.shape
            labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
            # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
            labels_per_image = labels[topk_indices]

            topk_indices = topk_indices // num_classes
            # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
            mask_pred = mask_pred[topk_indices]

            # if this is panoptic segmentation, we only keep the "thing" classes
            if self.panoptic_on:
                keep = torch.zeros_like(scores_per_image).bool()
                for i, lab in enumerate(labels_per_image):
                    keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]
                mask_pred = mask_pred[keep]

            result = Instances(image_size)
            # mask (before sigmoid)
            result.pred_masks = (mask_pred > 0).float()
            result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
            # Uncomment the following to get boxes from masks (this is slow)
            # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

            # calculate average mask prob
            mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
                    result.pred_masks.flatten(1).sum(1) + 1e-6)
            result.scores = scores_per_image * mask_scores_per_image
            result.pred_classes = labels_per_image
            return result

elif version == 'plop':
    # query freeze : True
    # query kd : True
    # prediction kd : True
    # query group attention mask : True


    class SoftDiceLoss(nn.Module):
        def __init__(self, weight=None, size_average=True):
            super(SoftDiceLoss, self).__init__()

        def forward(self, logits, targets):
            bs = targets.size(0)
            smooth = 1.0

            probs = F.sigmoid(logits)
            m1 = probs.view(bs, -1)
            m2 = targets.view(bs, -1)
            intersection = (m1 * m2)

            score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
            score = 1 - score.sum() / bs
            return score


    @META_ARCH_REGISTRY.register()
    class IncrementalMaskFormer(nn.Module):
        """
        Main class for mask classification semantic segmentation architectures.
        """

        @configurable
        def __init__(
                self,
                *,
                cfg: None,
                name: str,
                backbone: Backbone,
                sem_seg_head: nn.Module,
                criterion: nn.Module,
                num_queries: int,
                object_mask_threshold: float,
                overlap_threshold: float,
                metadata,
                size_divisibility: int,
                sem_seg_postprocess_before_inference: bool,
                pixel_mean: Tuple[float],
                pixel_std: Tuple[float],
                # inference
                semantic_on: bool,
                panoptic_on: bool,
                instance_on: bool,
                test_topk_per_image: int,
        ):
            """
            Args:
                backbone: a backbone module, must follow detectron2's backbone interface
                sem_seg_head: a module that predicts semantic segmentation from backbone features
                criterion: a module that defines the loss
                num_queries: int, number of queries
                object_mask_threshold: float, threshold to filter query based on classification score
                    for panoptic segmentation inference
                overlap_threshold: overlap threshold used in general inference for panoptic segmentation
                metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                    segmentation inference
                size_divisibility: Some backbones require the input height and width to be divisible by a
                    specific integer. We can use this to override such requirement.
                sem_seg_postprocess_before_inference: whether to resize the prediction back
                    to original input size before semantic segmentation inference or after.
                    For high-resolution dataset like Mapillary, resizing predictions before
                    inference will cause OOM error.
                pixel_mean, pixel_std: list or tuple with #channels element, representing
                    the per-channel mean and std to be used to normalize the input image
                semantic_on: bool, whether to output semantic segmentation prediction
                instance_on: bool, whether to output instance segmentation prediction
                panoptic_on: bool, whether to output panoptic segmentation prediction
                test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
            """
            super().__init__()
            self.cfg = copy.deepcopy(cfg)
            dataset, _, task, step = name.split('_')
            self.dataset = dataset
            self.task = task
            self.step = int(step)

            sem_seg_head.predictor.task = self.task
            sem_seg_head.predictor.step = self.step


            # freeze the old queries
            self.tmodel = None
            if self.step > 0:

                # get the teacher model
                if not self.cfg.MODEL.MODEL_OLD:
                    # load step t-1 teacher model
                    model_old_cfg = copy.deepcopy(cfg)
                    model_old_cfg.defrost()
                    model_old_cfg.MODEL.MODEL_OLD = True

                    model_old = build_model(model_old_cfg)
                    model_old.step = self.step - 1
                    model_old.sem_seg_head.predictor.step = self.step - 1
                    DetectionCheckpointer(model_old, save_dir='./cache/debug').resume_or_load(
                        model_old_cfg.MODEL.WEIGHTS
                    )
                    model_old.eval()
                    self.tmodel = {'model_old': model_old}
                else:
                    self.tmodel = None


            self.backbone = backbone
            self.sem_seg_head = sem_seg_head
            self.criterion = criterion
            self.num_queries = num_queries
            self.overlap_threshold = overlap_threshold
            self.object_mask_threshold = object_mask_threshold
            self.metadata = metadata
            if size_divisibility < 0:
                # use backbone size_divisibility if not set
                size_divisibility = self.backbone.size_divisibility
            self.size_divisibility = size_divisibility
            self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
            self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

            # additional args
            self.semantic_on = semantic_on
            self.instance_on = instance_on
            self.panoptic_on = panoptic_on
            self.test_topk_per_image = test_topk_per_image

            if not self.semantic_on:
                assert self.sem_seg_postprocess_before_inference

            self.dice_loss = SoftDiceLoss()

            self.params = {
                'collapse_channels': 'local',
                'pod_apply': 'all',
                'pod_factor': 1.,
                'prepro': 'pow',
                'spp_scales': [1, 2, 4],
                'pod_options': {"switch": {"after": {"extra_channels": "sum", "factor": 0.00001, "type": "local"}}},
                'use_pod_schedule': True,
            }

            self.num_base_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
            self.num_incre_clases = cfg.MODEL.MASK_FORMER.NUM_INCRE_CLASSES

        @property
        def num_classes(self):
                return self.num_base_classes + self.step*self.num_incre_clases


        @classmethod
        def from_config(cls, cfg):
            backbone = build_backbone(cfg)
            sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

            # Loss parameters:
            deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
            no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

            # loss weights
            class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
            dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
            mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

            # building criterion
            matcher = HungarianMatcher(
                cost_class=class_weight,
                cost_mask=mask_weight,
                cost_dice=dice_weight,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            )

            weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

            if deep_supervision:
                dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
                aux_weight_dict = {}
                for i in range(dec_layers - 1):
                    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
                weight_dict.update(aux_weight_dict)

            losses = ["labels", "masks"]

            dataset, seg, task, step = cfg.DATASETS.TRAIN[0].split('_')

            if dataset == 'ade20k':
                if seg == 'incrementalins':
                    from mask2former.data.datasets.register_ade20k_incremental_instance import tasks_ade_ins
                    training_classes = tasks_ade_ins[task][int(step)]
                else:
                    from mask2former.data.datasets.register_ade20k_incremental import tasks_ade
                    training_classes = tasks_ade[task][int(step)]
                cr_num_classes = len(training_classes)
            elif dataset == 'voc':
                from mask2former.data.datasets.register_voc_incremental import tasks_voc
                training_classes = tasks_voc[task][int(step)]
                cr_num_classes = len(training_classes)
            elif dataset == 'coco':
                from mask2former.data.datasets.register_coco_incremental_instance import tasks_coco_ins
                training_classes = tasks_coco_ins[task][int(step)]
                cr_num_classes = len(training_classes)
            else:
                print(f'no dataset {dataset}')
                raise ValueError('No dataset')

            num_base_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
            num_incre_classes = cfg.MODEL.MASK_FORMER.NUM_INCRE_CLASSES
            num_classes = num_base_classes + int(step)*num_incre_classes

            criterion = SetCriterion(
                num_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight,
                losses=losses,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
                oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
                importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            )

            return {
                "cfg": cfg,
                "name": cfg.DATASETS.TRAIN[0],
                "backbone": backbone,
                "sem_seg_head": sem_seg_head,
                "criterion": criterion,
                "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
                "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
                "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
                "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
                "sem_seg_postprocess_before_inference": (
                        cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                        or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                        or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
                ),
                "pixel_mean": cfg.MODEL.PIXEL_MEAN,
                "pixel_std": cfg.MODEL.PIXEL_STD,
                # inference
                "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
                "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
                "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
                "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            }

        @property
        def device(self):
            return self.pixel_mean.device

        def losskd(self, new_features, old_features):
            losskd = dict()

            wpod = [1000, 1000, 1]

            pod_old_features = []
            for k, v in old_features['res_features'].items():
                pod_old_features.append(v)
            pod_new_features = []
            for k, v in new_features['res_features'].items():
                pod_new_features.append(v)

            losskd[f'loss_kd_lpod'] = features_distillation(pod_old_features, pod_new_features, **self.params)*wpod[0]

            return losskd

        def detach_features(self, features):
            if isinstance(features, torch.Tensor):
                return features.detach()
            elif isinstance(features, dict):
                new_dict = dict()
                for k, v in features.items():
                    new_dict[k] = self.detach_features(v)
                return new_dict
            elif isinstance(features, list):
                return [self.detach_features(el) for el in features]
            elif features is None:
                return None
            else:
                raise ValueError("unknow type")

        def unify_features(self, base_features, old_features):
            for old_feat, base_feat in zip(old_features['predictions_class'],base_features['predictions_class']):
                old_feat['step0'] = base_feat['step0']
            for old_feat, base_feat in zip(old_features['predictions_mask'],base_features['predictions_mask']):
                old_feat['step0'] = base_feat['step0']
            for old_feat, base_feat in zip(old_features['predictions_query'],base_features['predictions_query']):
                old_feat['step0'] = base_feat['step0']
            return old_features

        def forward(self, batched_inputs, kd=False):
            if kd:
                return self.forward_old(batched_inputs)
            else:
                return self.forward_new(batched_inputs)

        def forward_old(self, batched_inputs):
            # forward path of the model
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            features = self.backbone(images.tensor)

            outputs = self.sem_seg_head(features)

            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                    mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                # if self.sem_seg_postprocess_before_inference:
                #     mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                #         mask_pred_result, image_size, height, width
                #     )
                #     mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    # if not self.sem_seg_postprocess_before_inference:
                    #     r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r

                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["instances"] = instance_r

            # semantic predictions
            sem_logits = self.semantic_inference(outputs['pred_logits'], outputs['pred_masks'])
            outputs['kd_features']['predictions_sem'] = sem_logits

            if self.semantic_on:
                # pesudo labels
                pesudo_labels = []
                bg_thr = 0.1
                for results in processed_results:
                    pesudo_label = {'labels':[], 'masks':[], 'gt_ignore':None}
                    sem_seg = results['sem_seg']
                    c, h, w = sem_seg.shape
                    fg_mask = sem_seg.max(0).values>bg_thr
                    sem_seg = sem_seg.argmax(0)

                    labels = torch.zeros((0)).to(sem_logits).to(torch.int64)
                    masks = torch.zeros((0,h,w)).to(sem_logits)
                    for class_idx in range(len(sem_seg)):
                        mask = sem_seg == class_idx
                        mask = (mask*fg_mask)[None]
                        if mask.any():
                            label = torch.tensor([class_idx + 1]).to(sem_seg)
                            labels = torch.cat([labels,label],0).to(torch.int64)
                            masks = torch.cat([masks, mask],0).to(torch.bool)

                    pesudo_label['labels'] = labels
                    pesudo_label['masks'] = masks
                    pesudo_labels.append(pesudo_label)
            else:
                pesudo_labels = []
                score_thr = 0.5
                for results in processed_results:
                    pesudo_label = {'labels': [], 'masks': [], 'gt_ignore': None}
                    scores = results['instances'].scores
                    scores_mask = scores>score_thr
                    pred_masks = results['instances'].pred_masks[scores_mask]
                    pred_classes = results['instances'].pred_classes[scores_mask]+1
                    pesudo_label['labels'] = pred_classes.to(torch.int64)
                    pesudo_label['masks'] = pred_masks.to(torch.bool)
                    pesudo_labels.append(pesudo_label)



            outputs['kd_features']['pesudo_labels'] = pesudo_labels

            # return the kd features
            kd_features = outputs['kd_features']
            kd_features['res_features'] = features
            return self.detach_features(kd_features)

        def forward_new(self, batched_inputs):
            # forward path of the model
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            features = self.backbone(images.tensor)

            outputs = self.sem_seg_head(features)

            # semantic predictions
            sem_logits = self.semantic_inference(outputs['pred_logits'], outputs['pred_masks'])
            outputs['kd_features']['predictions_sem'] = sem_logits

            if self.training:
                # mask classification target
                if "instances" in batched_inputs[0]:
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                    gt_ignores = [x["ignore"].to(self.device) for x in batched_inputs]
                    targets = self.prepare_targets(gt_instances, gt_ignores, images)
                else:
                    targets = None

                losses = dict()

                # cal the kd loss
                if self.training and self.tmodel is not None:
                    old_features = self.tmodel['model_old'](batched_inputs, kd=True)
                    pesudo_labels = old_features['pesudo_labels']
                    # from mask2former.data.datasets.register_ade20k_incremental import ADE20K_SEM_SEG
                    # plt.imshow(batched_inputs[0]['image'].permute(1, 2, 0).cpu())
                    # plt.show()
                    for target, pesudo_label in zip(targets, pesudo_labels):
                        target['labels'] = target['labels'] + self.tmodel['model_old'].num_classes
                        target['labels'] = torch.cat([target['labels'],pesudo_label['labels']],0)
                        target['masks'] = torch.cat([target['masks'], pesudo_label['masks']], 0)
                    new_features = outputs['kd_features']
                    new_features['res_features'] = features
                    losses.update(self.losskd(new_features, old_features))

                if 'kd_features' in outputs:
                    outputs.pop('kd_features')

                # bipartite matching-based loss
                losses.update(self.criterion(outputs, targets))

                for k in list(losses.keys()):
                    if k in self.criterion.weight_dict:
                        losses[k] *= self.criterion.weight_dict[k]
                    else:
                        # remove this loss if not specified in `weight_dict`
                        continue
                        # losses.pop(k)
                return losses
            else:
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )

                del outputs

                processed_results = []
                for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                        mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    processed_results.append({})

                    if self.sem_seg_postprocess_before_inference:
                        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            mask_pred_result, image_size, height, width
                        )
                        mask_cls_result = mask_cls_result.to(mask_pred_result)

                    # semantic segmentation inference
                    if self.semantic_on:
                        r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                        if not self.sem_seg_postprocess_before_inference:
                            r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                        processed_results[-1]["sem_seg"] = r

                    # panoptic segmentation inference
                    if self.panoptic_on:
                        panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["panoptic_seg"] = panoptic_r

                    # instance segmentation inference
                    if self.instance_on:
                        instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["instances"] = instance_r

                return processed_results

        def prepare_targets(self, targets, gt_ignores, images):
            h_pad, w_pad = images.tensor.shape[-2:]
            new_targets = []
            for targets_per_image, gt_ignore in zip(targets, gt_ignores):
                # pad gt
                gt_masks = targets_per_image.gt_masks
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype,
                                           device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
                new_targets.append(
                    {
                        "labels": targets_per_image.gt_classes,
                        "masks": padded_masks,
                        "gt_ignore": gt_ignore,
                    }
                )
            return new_targets

        def semantic_inference(self, mask_cls, mask_pred):

            mask_cls = F.softmax(mask_cls, dim=-1)[..., 1:]  # the first class is the background
            mask_pred = mask_pred.sigmoid()
            if len(mask_cls.shape)==2:
                semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            else:
                semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
            return semseg

        def panoptic_inference(self, mask_cls, mask_pred):
            scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
            mask_pred = mask_pred.sigmoid()

            keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[keep]
            cur_mask_cls = mask_cls[keep]
            cur_mask_cls = cur_mask_cls[:, :-1]

            cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

            h, w = cur_masks.shape[-2:]
            panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
            segments_info = []

            current_segment_id = 0

            if cur_masks.shape[0] == 0:
                # We didn't detect any mask :(
                return panoptic_seg, segments_info
            else:
                # take argmax
                cur_mask_ids = cur_prob_masks.argmax(0)
                stuff_memory_list = {}
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()
                    isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                    mask_area = (cur_mask_ids == k).sum().item()
                    original_area = (cur_masks[k] >= 0.5).sum().item()
                    mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        if mask_area / original_area < self.overlap_threshold:
                            continue

                        # merge stuff regions
                        if not isthing:
                            if int(pred_class) in stuff_memory_list.keys():
                                panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                                continue
                            else:
                                stuff_memory_list[int(pred_class)] = current_segment_id + 1

                        current_segment_id += 1
                        panoptic_seg[mask] = current_segment_id

                        segments_info.append(
                            {
                                "id": current_segment_id,
                                "isthing": bool(isthing),
                                "category_id": int(pred_class),
                            }
                        )

                return panoptic_seg, segments_info

        def instance_inference(self, mask_cls, mask_pred):
            # mask_pred is already processed to have the same shape as original input
            image_size = mask_pred.shape[-2:]

            # [Q, K]
            # scores = F.softmax(mask_cls, dim=-1)[:, :-1]
            scores = F.softmax(mask_cls, dim=-1)[:, 1:]
            num_queries, num_classes = scores.shape
            labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
            # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
            labels_per_image = labels[topk_indices]

            topk_indices = topk_indices // num_classes
            # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
            mask_pred = mask_pred[topk_indices]

            # if this is panoptic segmentation, we only keep the "thing" classes
            if self.panoptic_on:
                keep = torch.zeros_like(scores_per_image).bool()
                for i, lab in enumerate(labels_per_image):
                    keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]
                mask_pred = mask_pred[keep]

            result = Instances(image_size)
            # mask (before sigmoid)
            result.pred_masks = (mask_pred > 0).float()
            result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
            # Uncomment the following to get boxes from masks (this is slow)
            # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

            # calculate average mask prob
            mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
                    result.pred_masks.flatten(1).sum(1) + 1e-6)
            result.scores = scores_per_image * mask_scores_per_image
            result.pred_classes = labels_per_image
            return result


else:
    raise ValueError('Not a valid version')

