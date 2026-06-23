# Copyright (c) OpenMMLab. All rights reserved.
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Union
from torch import Tensor
from mmcv.cnn import Linear
from mmengine.model import constant_init
from mmengine.structures import InstanceData
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps, bbox_xyxy_to_cxcywh
from mmdet.utils import InstanceList, reduce_mean
from mmdet.models.losses import VarifocalLoss
from mmdet.registry import MODELS
from .dfine_head import DFINEHead, unimodal_distribution_focal_loss
from .grounding_rtdetr_head import ContrastiveHead
from ..layers.transformer.dfine_layers import bbox2distance
from ..utils import multi_apply

@MODELS.register_module()
class GroundingDFINEHead(DFINEHead):
    def __init__(
            self,
            txt_embeds: int = 512,
            **kwargs
    ):
        self.txt_embeds = txt_embeds
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize classification branch and regression branch of head."""
        num_wide_layers = self.num_pred_layer - self.eval_idx - 2
        scaled_dim = int(round(self.layer_scale * self.embed_dims))

        def _gen_cls_branch(img_embed: int, txt_embed: int):
            return ContrastiveHead(img_embed, txt_embed)

        def _gen_reg_branch(embed_dims: int, out_channels: int):
            reg_branch = []
            for _ in range(self.num_reg_fcs):
                reg_branch.append(Linear(embed_dims, embed_dims))
                reg_branch.append(nn.ReLU())
            reg_branch.append(Linear(embed_dims, out_channels))
            return nn.Sequential(*reg_branch)

        cls_branches = [
            _gen_cls_branch(self.embed_dims, self.txt_embeds)
            for _ in range(self.num_pred_layer - num_wide_layers - 1)
        ]
        cls_branches += [
            _gen_cls_branch(scaled_dim, self.txt_embeds)
            for _ in range(num_wide_layers)
        ]
        cls_branches += [
            _gen_cls_branch(self.embed_dims, self.txt_embeds)
        ]
        self.cls_branches = nn.ModuleList(cls_branches)

        reg_branches = [
            _gen_reg_branch(self.embed_dims, 4 * (self.reg_max + 1))
            for _ in range(self.num_pred_layer - num_wide_layers - 1)
        ]
        reg_branches += [
            _gen_reg_branch(scaled_dim, 4 * (self.reg_max + 1))
            for _ in range(num_wide_layers)
        ]
        reg_branches += [_gen_reg_branch(self.embed_dims, 4)]
        reg_branches += [_gen_reg_branch(self.embed_dims, 4)]  # pre_bbox_head
        self.reg_branches = nn.ModuleList(reg_branches)

    def init_weights(self) -> None:
        """Initialize weights of the Deformable DETR head."""
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)

    def _get_dn_targets_single(self, gt_instances: InstanceData,
                               img_meta: dict, dn_meta: Dict[str,
                                                             int]) -> tuple:
        """Get targets in denoising part for one image.

        Args:
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels of each image.
            - label_weights (Tensor]): Label weights of each image.
            - bbox_targets (Tensor): BBox targets of each image.
            - bbox_weights (Tensor): BBox weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        num_groups = dn_meta['num_denoising_groups']
        num_denoising_queries = dn_meta['num_denoising_queries']
        num_queries_each_group = int(num_denoising_queries / num_groups)
        device = gt_bboxes.device

        if len(gt_labels) > 0:
            t = torch.arange(len(gt_labels), dtype=torch.long, device=device)
            t = t.unsqueeze(0).repeat(num_groups, 1)
            pos_assigned_gt_inds = t.flatten()
            pos_inds = torch.arange(
                num_groups, dtype=torch.long, device=device)
            pos_inds = pos_inds.unsqueeze(1) * num_queries_each_group + t
            pos_inds = pos_inds.flatten()
        else:
            pos_inds = pos_assigned_gt_inds = \
                gt_bboxes.new_tensor([], dtype=torch.long)

        neg_inds = pos_inds + num_queries_each_group // 2

        # label targets
        labels = gt_bboxes.new_full((num_denoising_queries, ),
                                    dn_meta['num_class'],
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_denoising_queries)

        # bbox targets
        bbox_targets = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_weights = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        gt_bboxes_normalized = gt_bboxes / factor
        gt_bboxes_targets = bbox_xyxy_to_cxcywh(gt_bboxes_normalized)
        bbox_targets[pos_inds] = gt_bboxes_targets.repeat([num_groups, 1])

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    def _loss_dn_single(self, dn_cls_scores: Tensor, dn_bbox_preds: Tensor,
                        dn_bbox_corners: Optional[Tensor],
                        teacher: Optional[Tuple[Tensor, Tensor]],
                        initial_dn_bbox_preds: Optional[Tensor],
                        batch_gt_instances: InstanceList,
                        batch_img_metas: List[dict],
                        dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        """Denoising loss for outputs from a single decoder layer.

        Args:
            dn_cls_scores (Tensor): Classification scores of a single decoder
                layer in denoising part, has shape (bs, num_denoising_queries,
                cls_out_channels).
            dn_bbox_preds (Tensor): Regression outputs of a single decoder
                layer in denoising part. Each is a 4D-tensor with normalized
                coordinate format (cx, cy, w, h) and has shape
                (bs, num_denoising_queries, 4).
            dn_bbox_corners (Tensor):
                # TODO
            teacher (tuple[Tensor, Tensor]):
                # TODO
            initial_dn_bbox_preds (Tensor):
                # TODO
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        if dn_cls_scores.size(1) == 0:
            loss_cls = dn_cls_scores.new_tensor(0)
            loss_bbox = loss_iou = dn_bbox_preds.new_tensor(0)
            loss_fgl = loss_ddf = dn_bbox_corners.new_tensor(0) \
                if dn_bbox_corners is not None else None
            return loss_cls, loss_bbox, loss_iou, loss_fgl, loss_ddf

        if self.cached_dn_targets is None:
            cls_reg_targets = self.get_dn_targets(batch_gt_instances,
                                                  batch_img_metas, dn_meta)
            (labels_list, label_weights_list, bbox_targets_list,
             bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets
            labels = torch.cat(labels_list, 0)
            label_weights = torch.cat(label_weights_list, 0)
            bbox_targets = torch.cat(bbox_targets_list, 0)
            bbox_weights = torch.cat(bbox_weights_list, 0)

            # construct weighted avg_factor to match with the official DETR repo
            cls_avg_factor = \
                num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
            if self.sync_cls_avg_factor:
                cls_avg_factor = reduce_mean(
                    dn_bbox_preds.new_tensor([cls_avg_factor]))
            cls_avg_factor = max(cls_avg_factor, 1)

            # Compute the average number of gt boxes across all gpus, for
            # normalization purposes
            bbox_avg_factor = dn_bbox_preds.new_tensor([num_total_pos])
            bbox_avg_factor = torch.clamp(
                reduce_mean(bbox_avg_factor), min=1).item()

            self.cached_dn_targets = (labels, label_weights, bbox_targets,
                                      bbox_weights, num_total_pos,
                                      cls_avg_factor, bbox_avg_factor)
        else:
            # use cached dn targets
            (labels, label_weights, bbox_targets, bbox_weights, num_total_pos,
             cls_avg_factor, bbox_avg_factor) = self.cached_dn_targets

        # classification loss
        cls_scores = dn_cls_scores.reshape(-1, dn_cls_scores.shape[-1])

        if isinstance(self.loss_cls, VarifocalLoss):
            bg_class_ind = dn_cls_scores.shape[-1]
            pos_inds = ((labels >= 0)
                        & (labels < bg_class_ind)).nonzero().squeeze(1)
            cls_iou_targets = cls_scores.new_zeros(cls_scores.shape)
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
            pos_bbox_pred = dn_bbox_preds.reshape(-1, 4)[pos_inds]
            pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
            pos_labels = labels[pos_inds]
            cls_iou_targets[pos_inds, pos_labels] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True).type_as(cls_iou_targets)
            loss_cls = self.loss_cls(
                cls_scores, cls_iou_targets, avg_factor=cls_avg_factor)
        else:
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, dn_bbox_preds):
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = dn_bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=bbox_avg_factor)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=bbox_avg_factor)

        if dn_bbox_corners is None:
            return loss_cls, loss_bbox, loss_iou, None, None

        with_fgl_loss = self.fgl_loss_weight is not None
        with_dff_loss = self.loss_ld is not None and teacher is not None
        if not with_fgl_loss and not with_dff_loss:
            loss_fgl = loss_ddf = dn_bbox_corners.new_tensor(0)
            return loss_cls, loss_bbox, loss_iou, loss_fgl, loss_ddf

        bbox_pos_inds = torch.nonzero(
            bbox_weights.sum(-1) > 0, as_tuple=False).squeeze(-1).unique()
        pos_ious = bbox_overlaps(
            bboxes[bbox_pos_inds], bboxes_gt[bbox_pos_inds],
            is_aligned=True).detach()

        # distribution focal loss
        if with_fgl_loss:
            initial_dn_bbox_preds = initial_dn_bbox_preds.reshape(-1, 4)
            dn_bbox_corners = dn_bbox_corners.reshape(-1, 4, self.reg_max + 1)
            weight_targets = pos_ious.unsqueeze(-1).repeat(1, 4).reshape(-1)

            if self.cached_dn_fgl_targets is None:
                self.cached_dn_fgl_targets = bbox2distance(
                    initial_dn_bbox_preds[bbox_pos_inds],
                    bbox_cxcywh_to_xyxy(bbox_targets[bbox_pos_inds]),
                    self.reg_max, self.reg_scale, 0.5)
            (target_corners, weight_right,
             weight_left) = self.cached_dn_fgl_targets

            loss_fgl = self.fgl_loss_weight * unimodal_distribution_focal_loss(
                dn_bbox_corners[bbox_pos_inds].reshape(-1, self.reg_max + 1),
                target_corners,
                weight_right=weight_right,
                weight_left=weight_left,
                weight=weight_targets,
                avg_factor=bbox_avg_factor)
        else:
            loss_fgl = dn_bbox_corners.new_tensor(0)

        # vari KnowledgeDistillationKLDivLoss
        if with_dff_loss:
            teacher_scores, teacher_corners = teacher
            teacher_scores = teacher_scores.reshape(-1, teacher_scores.shape[-1])
            teacher_corners = teacher_corners.reshape(-1, self.reg_max + 1)
            dn_bbox_corners = dn_bbox_corners.reshape(-1, self.reg_max + 1)

            weight_targets_local = teacher_scores.sigmoid().max(dim=-1)[0]
            weight_targets_local[bbox_pos_inds] = \
                pos_ious.type_as(weight_targets_local)
            weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(
                1, 4).reshape(-1)

            loss_match_local = self.loss_ld(dn_bbox_corners, teacher_corners,
                                            weight_targets_local) * (
                                       self.reg_max + 1)

            mask = bbox_weights.bool().reshape(-1)
            num_total_bbox_pos = num_total_pos
            num_total_bbox_neg = bbox_weights.size(0) - num_total_bbox_pos
            loss_match_local1 = loss_match_local[mask].mean() \
                if num_total_bbox_pos > 0 else 0
            loss_match_local2 = loss_match_local[~mask].mean() \
                if num_total_bbox_neg > 0 else 0
            loss_ddf = (loss_match_local1 * self.num_pos +
                        loss_match_local2 * self.num_neg) / (
                               self.num_pos + self.num_neg)
        else:
            loss_ddf = dn_bbox_corners.new_tensor(0)

        return loss_cls, loss_bbox, loss_iou, loss_fgl, loss_ddf

    def _get_targets_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict) -> tuple:
        """Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_queries, 4].
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels of each image.
            - label_weights (Tensor]): Label weights of each image.
            - bbox_targets (Tensor): BBox targets of each image.
            - bbox_weights (Tensor): BBox weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        img_h, img_w = img_meta['img_shape']
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        num_bboxes = bbox_pred.size(0)
        # convert bbox_pred from xywh, normalized to xyxy, unnormalized
        bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_pred = bbox_pred * factor

        pred_instances = InstanceData(scores=cls_score, bboxes=bbox_pred)
        # assigner and sampler
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta)

        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long(), :]

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    cls_score.shape[-1],
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights = torch.zeros_like(bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights[pos_inds] = 1.0

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        pos_gt_bboxes_normalized = pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    def loss_by_feat_single(self, cls_scores: Tensor, bbox_preds: Tensor,
                            bbox_corners: Optional[Tensor],
                            teacher: Optional[Tuple[Tensor, Tensor]],
                            batch_match_indices: Optional[List[Tuple[Tensor,
                            Tensor]]],
                            initial_bbox_preds: Optional[Tensor],
                            merged_match_indices: List[Tuple[Tensor, Tensor]],
                            batch_gt_instances: InstanceList,
                            batch_img_metas: List[dict]) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images, has shape (bs, num_queries, cls_out_channels).
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape (bs, num_queries, 4).
            bbox_corners (Tensor):
                # TODO
            teacher (tuple[Tensor, Tensor]):
                # TODO
            batch_match_indices (list[tuple[Tensor, Tensor]]):
                # TODO
            initial_bbox_preds (Tensor):
                # TODO
            merged_match_indices (list[tuple[Tensor, Tensor]]):
                # TODO
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        num_imgs, num_queries, _ = cls_scores.shape
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         bbox_num_pos_list) = multi_apply(
            self._get_cls_targets_single,
            batch_match_indices,
            batch_gt_instances,
            batch_img_metas,
            num_queries=num_queries,
            device=bbox_preds.device)
        num_total_pos = sum(bbox_num_pos_list)
        num_total_neg = num_imgs * num_queries - num_total_pos
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, cls_scores.shape[-1])
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if isinstance(self.loss_cls, VarifocalLoss):
            bg_class_ind = cls_scores.shape[-1]
            pos_inds = ((labels >= 0)
                        & (labels < bg_class_ind)).nonzero().squeeze(1)
            cls_iou_targets = cls_scores.new_zeros(cls_scores.shape)
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
            pos_bbox_pred = bbox_preds.reshape(-1, 4)[pos_inds]
            pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
            pos_labels = labels[pos_inds]
            cls_iou_targets[pos_inds, pos_labels] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True).type_as(cls_iou_targets)
            loss_cls = self.loss_cls(
                cls_scores, cls_iou_targets, avg_factor=cls_avg_factor)
        else:
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        if num_queries not in self.cached_bbox_targets:
            if self.use_uni_set:
                (bbox_targets_list, bbox_weights_list,
                 bbox_num_pos_list) = multi_apply(
                     self._get_bbox_targets_single,
                     merged_match_indices,
                     batch_gt_instances,
                     batch_img_metas,
                     num_queries=num_queries,
                     device=bbox_preds.device)
                num_total_bbox_pos = sum(bbox_num_pos_list)
                bbox_targets = torch.cat(bbox_targets_list, 0)
            else:
                num_total_bbox_pos = num_total_pos

            bbox_weights = torch.cat(bbox_weights_list, 0)

            # Compute the average number of gt boxes across all gpus, for
            # normalization purposes
            bbox_avg_factor = bbox_preds.new_tensor([num_total_bbox_pos])
            bbox_avg_factor = torch.clamp(
                reduce_mean(bbox_avg_factor), min=1).item()

            self.cached_bbox_targets[num_queries] = (bbox_targets,
                                                     bbox_weights,
                                                     num_total_bbox_pos,
                                                     bbox_avg_factor)
        else:
            # use cached bbox targets
            (bbox_targets, bbox_weights, num_total_bbox_pos,
             bbox_avg_factor) = self.cached_bbox_targets[num_queries]

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            img_h, img_w, = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=bbox_avg_factor)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=bbox_avg_factor)

        if bbox_corners is None:
            return loss_cls, loss_bbox, loss_iou

        with_fgl_loss = self.fgl_loss_weight is not None
        with_dff_loss = self.loss_ld is not None and teacher is not None
        if not with_fgl_loss and not with_dff_loss:
            loss_fgl = loss_ddf = bbox_corners.new_tensor(0)
            return loss_cls, loss_bbox, loss_iou, loss_fgl, loss_ddf

        bbox_pos_inds = torch.nonzero(
            bbox_weights.sum(-1) > 0, as_tuple=False).squeeze(-1).unique()
        pos_ious = bbox_overlaps(
            bboxes[bbox_pos_inds], bboxes_gt[bbox_pos_inds],
            is_aligned=True).detach()

        # distribution focal loss
        if with_fgl_loss:
            initial_bbox_preds = initial_bbox_preds.reshape(-1, 4)
            bbox_corners = bbox_corners.reshape(-1, 4, self.reg_max + 1)
            weight_targets = pos_ious.unsqueeze(-1).repeat(1, 4).reshape(-1)

            if self.cached_fgl_targets is None:
                self.cached_fgl_targets = bbox2distance(
                    initial_bbox_preds[bbox_pos_inds],
                    bbox_cxcywh_to_xyxy(bbox_targets[bbox_pos_inds]),
                    self.reg_max, self.reg_scale, 0.5)
            target_corners, weight_right, weight_left = self.cached_fgl_targets

            loss_fgl = self.fgl_loss_weight * unimodal_distribution_focal_loss(
                bbox_corners[bbox_pos_inds].reshape(-1, self.reg_max + 1),
                target_corners,
                weight_right=weight_right,
                weight_left=weight_left,
                weight=weight_targets,
                avg_factor=bbox_avg_factor)
        else:
            loss_fgl = bbox_corners.new_tensor(0)

        # vari KnowledgeDistillationKLDivLoss
        if with_dff_loss:
            teacher_scores, teacher_corners = teacher
            teacher_scores = teacher_scores.reshape(-1, teacher_scores.shape[-1])
            teacher_corners = teacher_corners.reshape(-1, self.reg_max + 1)
            bbox_corners = bbox_corners.reshape(-1, self.reg_max + 1)

            weight_targets_local = teacher_scores.sigmoid().max(dim=-1)[0]
            weight_targets_local[bbox_pos_inds] = \
                pos_ious.type_as(weight_targets_local)
            weight_targets_local = \
                weight_targets_local.unsqueeze(-1).repeat(1, 4).reshape(-1)

            loss_match_local = self.loss_ld(bbox_corners, teacher_corners,
                                            weight_targets_local) * (
                                       self.reg_max + 1)

            mask = bbox_weights.bool().reshape(-1)
            num_total_bbox_neg = bbox_weights.size(0) - num_total_bbox_pos
            if self.num_pos is None:
                self.num_pos = (num_total_bbox_pos * 4 * 8 / num_imgs) ** 0.5
                self.num_neg = (num_total_bbox_neg * 4 * 8 / num_imgs) ** 0.5
            loss_match_local1 = loss_match_local[mask].mean() \
                if num_total_bbox_pos > 0 else 0
            loss_match_local2 = loss_match_local[~mask].mean() \
                if num_total_bbox_neg > 0 else 0
            loss_ddf = (loss_match_local1 * self.num_pos +
                        loss_match_local2 * self.num_neg) / (
                               self.num_pos + self.num_neg)
        else:
            loss_ddf = bbox_corners.new_tensor(0)

        return loss_cls, loss_bbox, loss_iou, loss_fgl, loss_ddf

    def _predict_by_feat_single(self,
                                cls_score: Tensor,
                                bbox_pred: Tensor,
                                img_meta: dict,
                                rescale: bool = True):
        assert len(cls_score) == len(bbox_pred)  # num_queries
        max_per_img = self.test_cfg.get('max_per_img', len(cls_score))
        img_shape = img_meta['img_shape']
        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            num_classes = cls_score.shape[-1]
            det_labels = indexes % num_classes
            bbox_index = indexes // num_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            assert img_meta.get('scale_factor') is not None
            det_bboxes /= det_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))

        results = InstanceData()
        results.bboxes = det_bboxes
        results.scores = scores
        results.labels = det_labels
        return results