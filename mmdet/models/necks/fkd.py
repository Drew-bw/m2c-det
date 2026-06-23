# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from typing import List
from mmengine.model import BaseModule
from mmdet.registry import MODELS

@MODELS.register_module()
class FKD(BaseModule):

    def __init__(
            self,
            channels: int = 256,
            weights: List = (0, 0, 0, 0),
            module_weights: List = (True, True, True, True),
            no_share_params: List = (1, 0.5, 0.5),
            low_order_module: dict = dict(
                type='MultiheadReduceAttn',
                embed_dims=256,
                num_heads=8,
                spatial_ratio=4),
    ) -> None:
        super().__init__()
        self.channels = channels
        level_weight = no_share_params[0]
        spatial_ratio = no_share_params[1]
        self.spa_t = no_share_params[2]
        self.chn_t = no_share_params[3]

        self.feat_weight =  weights[0] * level_weight if module_weights[0] else weights[0]
        self.spa_weight =  weights[1] * level_weight if module_weights[1] else weights[1]
        self.chn_weight = weights[2] * level_weight if module_weights[2] else weights[2]
        self.low_order_weight = weights[3] * level_weight if module_weights[3] else weights[3]

        self.feat_weight = self.feat_weight * 0.00001
        self.spa_weight = self.spa_weight * 0.00001
        self.low_order_weight = self.low_order_weight * 0.00001

        if self.low_order_weight > 0:
            low_order_module['spatial_ratio'] = spatial_ratio
            self.low_order_tea = MODELS.build(low_order_module)
            self.low_order_stu = MODELS.build(low_order_module)

    def get_spa_chn_attn_mask(self, student, teacher, p=1):

        b, c, h, w = student.shape
        # spatial
        spa_attn_stu = student.abs().pow(p).mean(dim=1).reshape(b, -1)
        spa_attn_tea = teacher.abs().pow(p).mean(dim=1).reshape(b, -1)
        spa_attn_stu = torch.softmax(spa_attn_stu / self.spa_t, dim=1) * h * w
        spa_attn_tea = torch.softmax(spa_attn_tea / self.spa_t, dim=1) * h * w
        spa_attn_stu = spa_attn_stu.reshape(b, 1, h, w)
        spa_attn_tea = spa_attn_tea.reshape(b, 1, h, w)

        # channel
        chn_attn_stu = student.abs().pow(p).reshape(b, c, -1).mean(dim=2)
        chn_attn_tea = teacher.abs().pow(p).reshape(b, c, -1).mean(dim=2)
        chn_attn_stu = torch.softmax(chn_attn_stu / self.chn_t, dim=1) * c
        chn_attn_tea = torch.softmax(chn_attn_tea / self.chn_t, dim=1) * c
        chn_attn_stu = chn_attn_stu.reshape(b, c, 1, 1)
        chn_attn_tea = chn_attn_tea.reshape(b, c, 1, 1)

        spa_attn_mask = spa_attn_tea.detach()
        chn_attn_mask = chn_attn_tea.detach()
        spa_chn_mask = torch.sqrt(spa_attn_mask * chn_attn_mask)

        return spa_attn_stu, chn_attn_stu, spa_attn_tea, chn_attn_tea, spa_chn_mask

    def forward(self, student, teacher, batch_data_samples):
        losses = dict()

        (spa_attn_s, chn_attn_s, spa_attn_t, chn_attn_t, spa_chn_mask
         ) = self.get_spa_chn_attn_mask(student, teacher, p=1)

        if self.feat_weight > 0:
            loss_feat = self.l2_loss(student, teacher, norm=False, mask=spa_chn_mask)
            losses['loss_feat'] = self.feat_weight * loss_feat

        if self.spa_weight > 0:
            loss_spatial = self.l2_loss(spa_attn_s, spa_attn_t, norm=False)
            losses['loss_spa'] = self.spa_weight * loss_spatial

        if self.chn_weight > 0:
            loss_channel = self.l2_loss(chn_attn_s, chn_attn_t, norm=False)
            losses['loss_chn'] = self.chn_weight * loss_channel

        if self.low_order_weight > 0:
            low_order_tea = self.low_order_tea(teacher)
            low_order_stu = self.low_order_stu(student)
            loss_low_order = self.l2_loss(low_order_stu, low_order_tea, norm=False)
            losses['loss_low_order'] = self.low_order_weight * loss_low_order

        return losses

    @staticmethod
    def l2_loss(s, t, norm=False, mask=None):

        b, c, h, w = s.shape

        if norm:
            s = F.normalize(s.reshape(b, -1)).reshape(b, c, h, w)
            t = F.normalize(t.reshape(b, -1)).reshape(b, c, h, w)

        if mask is not None:
            loss = torch.sum(mask * (s - t) ** 2) / b
        else:
            loss = torch.sum((s - t) ** 2) / b

        return loss