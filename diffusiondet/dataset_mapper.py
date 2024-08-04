# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T


__all__ = ["DiffusionDetDatasetMapper"]


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip())
    # ResizeShortestEdge
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))

    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


class DiffusionDetDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by DiffusionDet.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = [
                T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
                T.Resize(cfg.INPUT.CROP.SIZE),
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
                
            ]
        else:
            self.crop_gen = None

        self.tfm_gens = build_transform_gen(cfg, is_train)
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(str(self.tfm_gens), str(self.crop_gen))
        )

        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image_ir, image_ir_rgb, image_vi, visimage_bri, visimage_clr = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        
        # print(image_ir.shape, image_vi.shape, visimage_bri.shape, visimage_clr.shape)
        # print(image_vi.min(), image_vi.max())
        # utils.check_image_size(dataset_dict, image_ir)
        # utils.check_image_size(dataset_dict, image_ir_rgb)
        # utils.check_image_size(dataset_dict, image_vi)

        image = np.concatenate([image_ir, image_vi], axis=-1)
        image_shape = image.shape[:2]  # h, w
        # image_shape = image_ir.shape[1:]
        # print(image_shape)
        # Therefore it's important to use torch.Tensor.
        dataset_dict["ir"] = torch.as_tensor(np.ascontiguousarray(image_ir.transpose(2, 0, 1)).copy())
        dataset_dict["vi"] = torch.as_tensor(np.ascontiguousarray(image_vi.transpose(2, 0, 1)))
        # dataset_dict["ir"] = torch.as_tensor(np.ascontiguousarray(image_ir.transpose(2, 0, 1)))
        dataset_dict["ir_rgb"] = torch.as_tensor(np.ascontiguousarray(image_ir_rgb.transpose(2, 0, 1)))
        # dataset_dict["vi"] = torch.as_tensor(np.ascontiguousarray(image_vi.transpose(2, 0, 1)))
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image_vi.transpose(2, 0, 1)))
        dataset_dict["visimage_bri"] = torch.as_tensor(np.ascontiguousarray(visimage_bri.transpose(2, 0, 1)))
        dataset_dict["visimage_clr"] = torch.as_tensor(np.ascontiguousarray(visimage_clr.transpose(2, 0, 1)))
        # dataset_dict["ir"] = image_ir 
        # dataset_dict["ir_rgb"] = image_ir_rgb 
        # dataset_dict["vi"] = image_vi
        # dataset_dict["image"] = image_vi
        # dataset_dict["visimage_bri"] = visimage_bri 
        # dataset_dict["visimage_clr"] = visimage_clr
        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                obj
                # utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict
