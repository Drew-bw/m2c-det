# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Any, Union, Sequence
from torch import Tensor
import torch
import torch.nn as nn
from pathlib import Path
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmengine.structures import InstanceData
from mmengine.dist import get_dist_info
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptInstanceList
from mmdet.models.dense_heads.yolov8_head import gt_instances_preprocess
from .yolo_world_detector import YOLOWorldDetector
from mmdet.registry import MODELS


@MODELS.register_module()
class CMKDYOLOWorldDetector(YOLOWorldDetector):

    def __init__(
        self,
        teacher_config: Union[ConfigType, str, Path],
        teacher_ckpt: Optional[str] = None,
        eval_teacher: bool = True,
        neck_distill: OptConfigType = None,
        loss_ld: OptConfigType = None,
        **kwargs
    ) -> None:
        no_share_params = kwargs.pop('no_share_params') if kwargs.get('no_share_params') else [(True, )] * 5
        super().__init__(**kwargs)
        self.eval_teacher = eval_teacher
        teacher_model_cfg = Config.fromfile(teacher_config)['model']
        self.teacher_model = MODELS.build(teacher_model_cfg)
        if teacher_ckpt is not None:
            load_checkpoint(self.teacher_model, teacher_ckpt, map_location='cpu')

        if neck_distill is not None:
            self.neck_distill = nn.ModuleList()
            neck_channels = [128, 256, 512]
            for i, neck_channel in enumerate(neck_channels):
                neck_distill_ = neck_distill.copy()
                neck_distill_['no_share_params'] = no_share_params[i]
                if 'refine_module' in neck_distill_:
                    neck_distill_['refine_module']['in_channels'] = neck_channel
                self.neck_distill.append(MODELS.build(neck_distill_))

        if loss_ld is not None:
            self.loss_ld = MODELS.build(loss_ld)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        # import numpy as np
        # import cv2
        # img1 = batch_inputs[0, :3].permute(1, 2, 0).cpu().numpy() * 255
        # img1 = img1.astype(np.uint8).copy()
        # cv2.imwrite('a.jpg', img1)
        text_feats = self.language_model.forward_yolov5(batch_data_samples.pop('texts'))
        with torch.no_grad():
            img_feats_tea, text_feats = self.teacher_model.extract_feat(batch_inputs[:, 3:], text_feats)
            self.teacher_model.bbox_head.head_module.training = True
            outs_tea = self.teacher_model.bbox_head(img_feats_tea, text_feats)

        img_feats, text_feats = self.extract_feat(batch_inputs[:, :3], text_feats)
        outs = self.bbox_head(img_feats, text_feats)
        loss_inputs = outs + outs_tea + (batch_data_samples['bboxes_labels'],
                                         batch_data_samples['img_metas'])
        losses = self.loss_by_feat(*loss_inputs)

        if self.with_neck_distill:
            losses_neck = {}
            for i, (stu, tea) in enumerate(zip(img_feats, img_feats_tea)):
                loss_neck = self.neck_distill[i](stu, tea.detach(), batch_data_samples)
                for k, v in loss_neck.items():
                    if k not in losses_neck:
                        losses_neck[k] = [v]
                    else:
                        losses_neck[k].append(v)
            losses.update(losses_neck)

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        text_feats = self.language_model(batch_data_samples)
        x, text_feats = self.extract_feat(batch_inputs, text_feats)
        results_list = self.bbox_head.predict(
            x, text_feats, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def extract_feat(self, batch_inputs: Tensor, text_feats: Tensor):

        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x, text_feats)
        return x, text_feats

    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            bbox_dist_preds: Sequence[Tensor],
            cls_scores_tea: Sequence[Tensor],
            bbox_preds_tea: Sequence[Tensor],
            bbox_dist_preds_tea: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict]) -> dict:

        num_imgs = len(batch_img_metas)

        current_featmap_sizes = [
            cls_score.shape[2:] for cls_score in cls_scores
        ]
        # If the shape does not equal, generate new one
        if current_featmap_sizes != self.bbox_head.featmap_sizes_train:
            self.bbox_head.featmap_sizes_train = current_featmap_sizes

            mlvl_priors_with_stride = self.bbox_head.prior_generator.grid_priors(
                self.bbox_head.featmap_sizes_train,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device,
                with_stride=True)

            self.bbox_head.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.bbox_head.flatten_priors_train = torch.cat(mlvl_priors_with_stride, dim=0)
            self.bbox_head.stride_tensor = self.bbox_head.flatten_priors_train[..., [2]]

        # gt info
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]  # xyxy
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()
        num_classes = cls_scores[0].shape[1]
        # pred info
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, num_classes)
            for cls_pred in cls_scores
        ]
        flatten_pred_bboxes = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        # (bs, n, 4 * reg_max)
        flatten_pred_dists = [
            bbox_pred_org.reshape(num_imgs, -1, self.bbox_head.head_module.reg_max * 4)
            for bbox_pred_org in bbox_dist_preds
        ]

        flatten_dist_preds = torch.cat(flatten_pred_dists, dim=1)
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_pred_bboxes = self.bbox_head.bbox_coder.decode(
            self.bbox_head.flatten_priors_train[..., :2], flatten_pred_bboxes,
            self.bbox_head.stride_tensor[..., 0])

        assigned_result = self.bbox_head.assigner(
            (flatten_pred_bboxes.detach()).type(gt_bboxes.dtype),
            flatten_cls_preds.detach().sigmoid(), self.bbox_head.flatten_priors_train,
            gt_labels, gt_bboxes, pad_bbox_flag)

        assigned_bboxes = assigned_result['assigned_bboxes']
        assigned_scores = assigned_result['assigned_scores']
        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']

        assigned_scores_sum = assigned_scores.sum().clamp(min=1)

        loss_cls = self.bbox_head.loss_cls(flatten_cls_preds, assigned_scores).sum()
        loss_cls /= assigned_scores_sum

        # rescale bbox
        assigned_bboxes /= self.bbox_head.stride_tensor
        flatten_pred_bboxes /= self.bbox_head.stride_tensor

        # select positive samples mask
        num_pos = fg_mask_pre_prior.sum()
        if num_pos > 0:
            # when num_pos > 0, assigned_scores_sum will >0, so the loss_bbox
            # will not report an error
            # iou loss
            prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(
                flatten_pred_bboxes, prior_bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, prior_bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(assigned_scores.sum(-1),
                                              fg_mask_pre_prior).unsqueeze(-1)
            loss_bbox = self.bbox_head.loss_bbox(
                pred_bboxes_pos, assigned_bboxes_pos,
                weight=bbox_weight) / assigned_scores_sum

            # dfl loss
            pred_dist_pos = flatten_dist_preds[fg_mask_pre_prior]
            assigned_ltrb = self.bbox_head.bbox_coder.encode(
                self.bbox_head.flatten_priors_train[..., :2] / self.bbox_head.stride_tensor,
                assigned_bboxes,
                max_dis=self.bbox_head.head_module.reg_max - 1,
                eps=0.01)
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, prior_bbox_mask).reshape([-1, 4])
            loss_dfl = self.bbox_head.loss_dfl(pred_dist_pos.reshape(
                -1, self.bbox_head.head_module.reg_max),
                assigned_ltrb_pos.reshape(-1),
                weight=bbox_weight.expand(-1, 4).reshape(-1),
                avg_factor=assigned_scores_sum)

            # TODO
            if self.with_loss_ld:
                loss_ld = self.loss_ld()
            else:
                loss_ld = flatten_pred_bboxes.sum() * 0
        else:
            loss_bbox = flatten_pred_bboxes.sum() * 0
            loss_dfl = flatten_pred_bboxes.sum() * 0
            loss_ld = flatten_pred_bboxes.sum() * 0

        if self.bbox_head.world_size == -1:
            _, world_size = get_dist_info()
        else:
            world_size = self.bbox_head.world_size
        return dict(
            loss_cls=loss_cls * num_imgs * world_size,
            loss_bbox=loss_bbox * num_imgs * world_size,
            loss_dfl=loss_dfl * num_imgs * world_size,
            loss_ld=loss_ld * num_imgs * world_size)

    @property
    def with_neck_distill(self) -> bool:
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck_distill') and self.neck_distill is not None

    @property
    def with_loss_ld(self) -> bool:
        """bool: whether the detector has a neck"""
        return hasattr(self, 'loss_ld') and self.loss_ld is not None

    def cuda(self, device: Optional[str] = None) -> nn.Module:
        """Since teacher_model is registered as a plain object, it is necessary
        to put the teacher model to cuda when calling ``cuda`` function."""
        self.teacher_model.cuda(device=device)
        return super().cuda(device=device)

    def to(self, device: Optional[str] = None) -> nn.Module:
        """Since teacher_model is registered as a plain object, it is necessary
        to put the teacher model to other device when calling ``to``
        function."""
        self.teacher_model.to(device=device)
        return super().to(device=device)

    def train(self, mode: bool = True) -> None:
        """Set the same train mode for teacher and student model."""
        if self.eval_teacher:
            self.teacher_model.train(False)
        else:
            self.teacher_model.train(mode)
        super().train(mode)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute, i.e. self.name = value

        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
        if name == 'teacher_model':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)