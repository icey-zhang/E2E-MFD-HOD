# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import math
import random
from typing import List
from collections import namedtuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# from detectron2.layers import batched_nms, fusion
from detectron2.layers import batched_nms, fusion
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess

from detectron2.structures import Boxes, ImageList, Instances

from .loss0305 import SetCriterionDynamicK, HungarianMatcherDynamicK,DetcropPixelLoss
from .head import DynamicHead
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.misc import nested_tensor_from_tensor_list
from PIL import Image
import time

####论文对齐，损失函数前面的系数是10
#### loss0305

__all__ = ["DiffusionDet"]

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


@META_ARCH_REGISTRY.register()
class DiffusionDet(nn.Module):
    """
    Implement DiffusionDet
    """

    def __init__(self, cfg, return_fusion=False):
        super().__init__()
        self.return_fusion = return_fusion

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.DiffusionDet.NUM_CLASSES
        self.num_proposals = cfg.MODEL.DiffusionDet.NUM_PROPOSALS
        self.hidden_dim = cfg.MODEL.DiffusionDet.HIDDEN_DIM
        self.num_heads = cfg.MODEL.DiffusionDet.NUM_HEADS

        # Build Backbone.
        self.backbone = build_backbone(cfg)
        self.fusion = fusion.FusionNet(block_num=3, feature_out=False)
        # self.fusion = fusion.FusionNet(block_num=3, feature_out=False)
        # 加载预训练的分类模型
        # self.Illumination_classifier = fusion.Illumination_classifier(input_channels=3)
        # self.Illumination_classifier.load_state_dict(torch.load("/home/zjq/tolinux/code_M3FD0215/model_data/best_cls.pth"))
        # self.Illumination_classifier = self.Illumination_classifier.to(self.device).eval()
        # self.fusion.eval()
        self.size_divisibility = self.backbone.size_divisibility

        # build diffusion
        timesteps = 1000
        sampling_timesteps = cfg.MODEL.DiffusionDet.SAMPLE_STEP
        self.objective = 'pred_x0'
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.alpha = 0.
        self.beta = 1.

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.scale = cfg.MODEL.DiffusionDet.SNR_SCALE
        self.box_renewal = True
        self.use_ensemble = True

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # Build Dynamic Head.
        self.head = DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())
        #  Loss parameters:
        class_weight = cfg.MODEL.DiffusionDet.CLASS_WEIGHT
        giou_weight = cfg.MODEL.DiffusionDet.GIOU_WEIGHT
        l1_weight = cfg.MODEL.DiffusionDet.L1_WEIGHT
        no_object_weight = cfg.MODEL.DiffusionDet.NO_OBJECT_WEIGHT
        self.deep_supervision = cfg.MODEL.DiffusionDet.DEEP_SUPERVISION
        self.use_focal = cfg.MODEL.DiffusionDet.USE_FOCAL
        self.use_fed_loss = cfg.MODEL.DiffusionDet.USE_FED_LOSS
        self.use_nms = cfg.MODEL.DiffusionDet.USE_NMS

        # Build Criterion.
        matcher = HungarianMatcherDynamicK(
            cfg=cfg, cost_class=class_weight, cost_bbox=l1_weight, cost_giou=giou_weight, use_focal=self.use_focal
        )
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight}
        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes"]

        self.criterion = SetCriterionDynamicK(
            cfg=cfg, num_classes=self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight,
            losses=losses, use_focal=self.use_focal, )
        self.criterion_fuse = DetcropPixelLoss()

        pixel_mean1 = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_mean = torch.Tensor([0., 0., 0., 0.]).to(self.device).view(4, 1, 1)
        # pixel_mean = torch.cat([pixel_mean, pixel_mean], dim=0)
        pixel_std1 = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor([1., 1., 1., 1.]).to(self.device).view(4, 1, 1)
        # pixel_std = torch.cat([pixel_std, pixel_std], dim=0)
        # self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.normalizer1 = lambda x: (x - pixel_mean1) / pixel_std1
        self.to(self.device)

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, backbone_feats, images_whwh, x, t, x_self_cond=None, clip_x_start=False):
        x_boxes = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x_boxes = ((x_boxes / self.scale) + 1) / 2
        x_boxes = box_cxcywh_to_xyxy(x_boxes)
        x_boxes = x_boxes * images_whwh[:, None, :]
        outputs_class, outputs_coord = self.head(backbone_feats, x_boxes, t, None)

        x_start = outputs_coord[-1]  # (batch, num_proposals, 4) predict boxes: absolute coordinates (x1, y1, x2, y2)
        x_start = x_start / images_whwh[:, None, :]
        x_start = box_xyxy_to_cxcywh(x_start)
        x_start = (x_start * 2 - 1.) * self.scale
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start), outputs_class, outputs_coord

    @torch.no_grad()
    def ddim_sample(self, batched_inputs, backbone_feats, images_whwh, images, clip_denoised=True, do_postprocess=True):
        batch = images_whwh.shape[0]
        shape = (batch, self.num_proposals, 4)
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=self.device)

        ensemble_score, ensemble_label, ensemble_coord = [], [], []
        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None

            preds, outputs_class, outputs_coord = self.model_predictions(backbone_feats, images_whwh, img, time_cond,
                                                                         self_cond, clip_x_start=clip_denoised)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start

            if self.box_renewal:  # filter
                score_per_image, box_per_image = outputs_class[-1][0], outputs_coord[-1][0]
                threshold = 0.5
                score_per_image = torch.sigmoid(score_per_image)
                value, _ = torch.max(score_per_image, -1, keepdim=False)
                keep_idx = value > threshold
                num_remain = torch.sum(keep_idx)

                pred_noise = pred_noise[:, keep_idx, :]
                x_start = x_start[:, keep_idx, :]
                img = img[:, keep_idx, :]
            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            if self.box_renewal:  # filter
                # replenish with randn boxes
                img = torch.cat((img, torch.randn(1, self.num_proposals - num_remain, 4, device=img.device)), dim=1)
            if self.use_ensemble and self.sampling_timesteps > 1:
                box_pred_per_image, scores_per_image, labels_per_image = self.inference(outputs_class[-1],
                                                                                        outputs_coord[-1],
                                                                                        images.image_sizes)
                ensemble_score.append(scores_per_image)
                ensemble_label.append(labels_per_image)
                ensemble_coord.append(box_pred_per_image)

        if self.use_ensemble and self.sampling_timesteps > 1:
            box_pred_per_image = torch.cat(ensemble_coord, dim=0)
            scores_per_image = torch.cat(ensemble_score, dim=0)
            labels_per_image = torch.cat(ensemble_label, dim=0)
            if self.use_nms:
                keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                box_pred_per_image = box_pred_per_image[keep]
                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]

            result = Instances(images.image_sizes[0])
            result.pred_boxes = Boxes(box_pred_per_image)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results = [result]
        else:
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)
        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def change_hsv2rgb(self, fus_img, visimage_clr):
        bri = fus_img.detach().cpu().numpy() * 255
        bri = bri.reshape([fus_img.size()[2], fus_img.size()[3]])
        bri = np.where(bri < 0, 0, bri)
        bri = np.where(bri > 255, 255, bri)
        im1 = Image.fromarray(bri.astype(np.uint8))
        # bri = (visimage_bri*255).cpu().numpy().squeeze(0).transpose(1, 2, 0)
        clr = visimage_clr.cpu().numpy().squeeze().transpose(1, 2, 0)
        clr = np.concatenate((clr, bri.reshape(fus_img.size()[2], fus_img.size()[3], 1)), axis=2)

        clr[:, :, 2] = im1
        clr = cv2.cvtColor(clr.astype(np.uint8), cv2.COLOR_HSV2RGB) #zjq
        # clr = cv2.cvtColor(clr.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
        # plt.subplot(233)
        # plt.axis('off')
        # plt.imshow(clr)
        # plt.savefig("rgb.png")
        return clr

    def fusion_run(self, batched_inputs):
        images, vi_rgb, ir_rgb, ir_image, vi_image, visimage_bri, visimage_clr, images_whwh = self.preprocess_image(
            batched_inputs, self.training)
        
        vi_rgb = vi_rgb.tensor
        visimage_bri = visimage_bri.tensor / 255.
        visimage_clr = visimage_clr.tensor
        ir_image = ir_image.tensor
        vi_image = vi_image.tensor
        ir_rgb = ir_rgb.tensor
        inputs = torch.cat([ir_image, vi_image], dim=1)
        start = time.time()
        src = self.backbone(vi_rgb)          #vi_image是/255的，vi_rgb是用的normalizer1
        src1 = self.backbone(ir_rgb) #3个通道     ir_rgb是用的normalizer1   ir_image是/255的
        # src = self.backbone(ir_rgb)
        # src = self.backbone(inputs)
        features = list()
        for i in self.in_features:
            feature = src[i]+src1[i]
            features.append(feature)
        _, res_weight = self.fusion(features,inputs) #这个测试需要修改
        end = time.time()
        time_per_img = end-start
        fus_img = res_weight[:, 0, :, :] * ir_image + res_weight[:, 1, :, :] * visimage_bri #/ 255.
        # plt.imshow((visimage_bri*255).cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8))
        # plt.savefig("visimage_bri.png")
        # plt.imshow((ir_image*255).cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8))
        # plt.savefig("ir_image.png")
        # plt.imshow((vi_image*255).cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8))
        # plt.savefig("vi_image.png")
        ####
        # plt.imshow((fus_img*255).cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8))
        # plt.savefig("fus_img.png")
        fusion_img = self.change_hsv2rgb(fus_img, visimage_clr)
        ####
        # fusion_img = fusion_img / 255.
        # fusion_img = torch.from_numpy(fusion_img.transpose(2, 0, 1))[None].type(torch.FloatTensor)
        return fus_img,fusion_img,time_per_img


    def forward(self, batched_inputs, do_postprocess=True):
        images, vi_rgb, ir_rgb, ir_image, vi_image, visimage_bri, visimage_clr, images_whwh = self.preprocess_image(batched_inputs, self.training)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        vi_rgb = vi_rgb.tensor
        ir_rgb = ir_rgb.tensor
        visimage_bri = visimage_bri.tensor / 255.
        visimage_clr = visimage_clr.tensor
        ir_image = ir_image.tensor
        vi_image = vi_image.tensor
        inputs = torch.cat([ir_image, vi_image], dim=1)
        # plt.subplot(231)
        # plt.axis('off')
        # plt.imshow((ir_image*255).cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8))
        # # plt.savefig("ir_image.png")
        # plt.subplot(232)
        # plt.axis('off')
        # plt.imshow((visimage_bri*255).cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8))
        # plt.subplot(232)
        # plt.axis('off')
        # plt.imshow((vi_image*255).cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8))
        # plt.savefig("vi_image.png")
        # 使用预训练的分类模型，得到可见光图片属于白天还是夜晚的概率
        # pred = self.Illumination_classifier(vi_image)
        # day_p = pred[:, 0] #17.71
        # night_p = pred[:, 1] #0
        # vis_weight = day_p / (day_p + night_p)
        # inf_weight = 1 - vis_weight
        # if vis_weight == 1:
        #     plt.text(1500, 900, "day", fontsize=12, color='red')
        # else:
        #     plt.text(1500, 900, "night", fontsize=12, color='red')
        vis_weight = None
        inf_weight = None
        #ir_rgb就是近红外图像
        src = self.backbone(vi_rgb)          #vi_image是/255的，vi_rgb是用的normalizer1
        src1 = self.backbone(ir_rgb) #3个通道     ir_rgb是用的normalizer1   ir_image是/255的
        # src = self.backbone(ir_rgb)
        # src = self.backbone(inputs)
        features = list()
        for i in self.in_features:
            feature = src[i]+src1[i]
            features.append(feature)
            # if i == 'p2':
            #     feature_for_fusion 
        # Prepare Proposals.
        # _, res_weight = self.fusion(inputs)
        _, res_weight = self.fusion(features,inputs)#,features[2])
        fus_img = res_weight[:, 0:1, :, :] * ir_image + res_weight[:, 1:, :, :] * visimage_bri#visimage_bri / 255.
        # w1 = res_weight[:, 0, :, :].detach().cpu().numpy()[0]
        # w2 = res_weight[:, 1, :, :].detach().cpu().numpy()[0]
        # plt.subplot(234)
        # plt.axis('off')
        # plt.imshow(w1, cmap='Greys', interpolation='nearest')
        # # plt.savefig("w1.png")
        # plt.subplot(235)
        # plt.axis('off')
        # plt.imshow(w2, cmap='Greys', interpolation='nearest')
        # plt.savefig("w2.png")
        # plt.imshow((fus_img*255).detach().cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8))
        # plt.savefig("fus_img.png")
        #
        
        # fusion_img = fusion_img / 255.
        # fusion_img = torch.from_numpy(fusion_img.transpose(2, 0, 1))[None].type(torch.FloatTensor)
        

        if not self.training:
            if self.return_fusion:
                fusion_img = self.change_hsv2rgb(fus_img, visimage_clr)
                return _, fusion_img
            results = self.ddim_sample(batched_inputs, features, images_whwh, images)
            return results

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets, x_boxes, noises, t, mask_list = self.prepare_targets(gt_instances)
            mask = torch.stack(mask_list).unsqueeze(0).to(self.device)
            pad1 = int((fus_img.shape[-2]-mask.shape[-2])/2)
            pad2 = int((fus_img.shape[-1]-mask.shape[-1])/2)
            mask = F.pad(mask, (pad2,pad2,pad1,pad1))
            # plt.subplot(236)
            # plt.axis('off')
            # plt.imshow(mask.detach().cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8))
            # # plt.savefig("mask.png")
            # plt.savefig("rgb77.png")
            # if vis_weight != None:
            #     SSIM_loss,grad_loss,pixel_loss,ilu_loss = self.criterion_fuse(fus_img, visimage_bri, ir_image, mask, inf_weight,vis_weight)
            # else:
            SSIM_loss,grad_loss,pixel_loss = self.criterion_fuse(fus_img, visimage_bri, ir_image, mask, inf_weight,vis_weight)

            #### 做mask的可视化
            # plt.imshow(mask.cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8))
            # plt.savefig("mask.png")
            # plt.imshow((vi_image*255).cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8))
            # plt.savefig("vi_image.png")
            # plt.imshow((ir_image*255).cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8))
            # plt.savefig("ir_image.png")
            # 将叠加图像叠加到背景图像上
            # background_image = (vi_image*255).cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8)
            # overlay_image = (mask*255).cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8)
            # alpha = 0.5  # 设置叠加图像的透明度
            # x_offset = 0  # 设置叠加图像的水平偏移量
            # y_offset = 0  # 设置叠加图像的垂直偏移量
            # background_image[y_offset:y_offset+overlay_image.shape[0], x_offset:x_offset+overlay_image.shape[1]] = \
            #     alpha * overlay_image + (1 - alpha) * background_image[y_offset:y_offset+overlay_image.shape[0], x_offset:x_offset+overlay_image.shape[1]]
            # # 使用 Matplotlib 显示叠加后的图像
            # plt.imshow(background_image)
            # plt.axis('off')
            # plt.savefig("vi_mask.png")

            t = t.squeeze(-1)
            x_boxes = x_boxes * images_whwh[:, None, :]

            outputs_class, outputs_coord = self.head(features, x_boxes, t, None)
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

            if self.deep_supervision:
                output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b}
                                         for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
            
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            loss_dict['SSIM_loss'] = 10*SSIM_loss 
            loss_dict['grad_loss'] = 10*grad_loss 
            loss_dict['pixel_loss'] = 10*pixel_loss 
            loss_dict_new = {}
            loss_object = loss_dict['loss_ce'] + loss_dict['loss_bbox'] + loss_dict['loss_giou'] 
            loss_object += loss_dict['loss_ce_0'] + loss_dict['loss_bbox_0'] + loss_dict['loss_giou_0'] 
            loss_object += loss_dict['loss_ce_1'] + loss_dict['loss_bbox_1'] + loss_dict['loss_giou_1'] 
            loss_object += loss_dict['loss_ce_2'] + loss_dict['loss_bbox_2'] + loss_dict['loss_giou_2']
            loss_object += loss_dict['loss_ce_3'] + loss_dict['loss_bbox_3'] + loss_dict['loss_giou_3']
            loss_object += loss_dict['loss_ce_4'] + loss_dict['loss_bbox_4'] + loss_dict['loss_giou_4']
            loss_dict_new['task_ob'] = loss_object 
            loss_dict_new['task_fu'] = loss_dict['SSIM_loss'] + loss_dict['grad_loss'] + loss_dict['pixel_loss'] 
            # if vis_weight != None:
            #     loss_dict['ilu_loss'] = 10*ilu_loss 
            
            # SSIM_loss,grad_loss,pixel_loss
            # return loss_dict_new
            # if vis_weight != None:
            #     loss_dict['ilu_loss'] = 10*ilu_loss 
            
            # SSIM_loss,grad_loss,pixel_loss
            # print("task_ob:",loss_dict_new['task_ob'])
            # print("task_fu:",loss_dict_new['task_fu'])
            return loss_dict_new

    def prepare_diffusion_repeat(self, gt_boxes):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        noise = torch.randn(self.num_proposals, 4, device=self.device)

        num_gt = gt_boxes.shape[0]
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=torch.float, device=self.device)
            num_gt = 1

        num_repeat = self.num_proposals // num_gt  # number of repeat except the last gt box in one image
        repeat_tensor = [num_repeat] * (num_gt - self.num_proposals % num_gt) + [num_repeat + 1] * (
                self.num_proposals % num_gt)
        assert sum(repeat_tensor) == self.num_proposals
        random.shuffle(repeat_tensor)
        repeat_tensor = torch.tensor(repeat_tensor, device=self.device)

        gt_boxes = (gt_boxes * 2. - 1.) * self.scale
        x_start = torch.repeat_interleave(gt_boxes, repeat_tensor, dim=0)

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        diff_boxes = box_cxcywh_to_xyxy(x)

        return diff_boxes, noise, t

    def prepare_diffusion_concat(self, gt_boxes):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        noise = torch.randn(self.num_proposals, 4, device=self.device)

        num_gt = gt_boxes.shape[0]
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=torch.float, device=self.device)
            num_gt = 1

        if num_gt < self.num_proposals:
            box_placeholder = torch.randn(self.num_proposals - num_gt, 4,
                                          device=self.device) / 6. + 0.5  # 3sigma = 1/2 --> sigma: 1/6
            box_placeholder[:, 2:] = torch.clip(box_placeholder[:, 2:], min=1e-4)
            x_start = torch.cat((gt_boxes, box_placeholder), dim=0)
        elif num_gt > self.num_proposals:
            select_mask = [True] * self.num_proposals + [False] * (num_gt - self.num_proposals)
            random.shuffle(select_mask)
            x_start = gt_boxes[select_mask]
        else:
            x_start = gt_boxes

        x_start = (x_start * 2. - 1.) * self.scale

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        diff_boxes = box_cxcywh_to_xyxy(x)

        return diff_boxes, noise, t

    def prepare_targets(self, targets):
        new_targets = []
        diffused_boxes = []
        noises = []
        ts = []
        mask_list = []
        for targets_per_image in targets:
            target = {}
            h, w = targets_per_image.image_size
            mask = torch.zeros(h,w)
            # ones_mask = torch.ones(h,w)
            detect_box = targets_per_image.gt_boxes.tensor
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            #构建mask
            for k in range(detect_box.shape[0]):
                mask[int(detect_box[k][1]):int(detect_box[k][3]),int(detect_box[k][0]):int(detect_box[k][2])]=1#.data.copy_(ones_mask[int(detect_box[k][1]):int(detect_box[k][3]),int(detect_box[k][0]):int(detect_box[k][2])])
            mask_list.append(mask)
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            d_boxes, d_noise, d_t = self.prepare_diffusion_concat(gt_boxes)
            diffused_boxes.append(d_boxes)
            noises.append(d_noise)
            ts.append(d_t)
            target["labels"] = gt_classes.to(self.device)
            target["boxes"] = gt_boxes.to(self.device)
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device)
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            target["area"] = targets_per_image.gt_boxes.area().to(self.device)
            new_targets.append(target)

        return new_targets, torch.stack(diffused_boxes), torch.stack(noises), torch.stack(ts),mask_list

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        if self.use_focal or self.use_fed_loss:
            scores = torch.sigmoid(box_cls)
            labels = torch.arange(self.num_classes, device=self.device). \
                unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)

            for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, box_pred, image_sizes
            )):
                result = Instances(image_size)
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False)
                labels_per_image = labels[topk_indices]
                box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
                box_pred_per_image = box_pred_per_image[topk_indices]

                if self.use_ensemble and self.sampling_timesteps > 1:
                    return box_pred_per_image, scores_per_image, labels_per_image

                if self.use_nms:
                    keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]

                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        else:
            # For each box we assign the best class or the second best if the best on is `no_object`.
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

            for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, labels, box_pred, image_sizes
            )):
                if self.use_ensemble and self.sampling_timesteps > 1:
                    return box_pred_per_image, scores_per_image, labels_per_image

                if self.use_nms:
                    keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]
                result = Instances(image_size)
                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        return results

    def preprocess_image(self, batched_inputs, training):
        """
        Normalize, pad and batch the input images.
        """
        if training:
            images = [self.normalizer1(x["image"].to(self.device)) for x in batched_inputs]
            ir_rgb = [self.normalizer1(x["ir_rgb"].to(self.device)) for x in batched_inputs]
            ir_image = [x["ir"].to(self.device) / 255. for x in batched_inputs]
            vi_image = [x["vi"].to(self.device) / 255. for x in batched_inputs]
            visimage_bri = [x["visimage_bri"].to(self.device) for x in batched_inputs]
            visimage_clr = [x["visimage_clr"].to(self.device) for x in batched_inputs]
        else:
            images = [self.normalizer1(x["image"].to(self.device)) for x in batched_inputs]
            ir_rgb = [self.normalizer1(x["ir_rgb"].to(self.device)) for x in batched_inputs]
            ir_image = [x["ir"].to(self.device) / 255. for x in batched_inputs]
            vi_image = [x["vi"].to(self.device) / 255. for x in batched_inputs]
            visimage_bri = [x["visimage_bri"].to(self.device) for x in batched_inputs]
            visimage_clr = [x["visimage_clr"].to(self.device) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)
        vi_rgb = images
        ir_rgb = ImageList.from_tensors(ir_rgb, self.size_divisibility)
        ir_image = ImageList.from_tensors(ir_image, self.size_divisibility)
        vi_image = ImageList.from_tensors(vi_image, self.size_divisibility)
        visimage_bri = ImageList.from_tensors(visimage_bri, self.size_divisibility)
        visimage_clr = ImageList.from_tensors(visimage_clr, self.size_divisibility)

        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)

        return images, vi_rgb, ir_rgb, ir_image, vi_image, visimage_bri, visimage_clr, images_whwh
