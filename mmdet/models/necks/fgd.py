# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from typing import List
from mmengine.model import BaseModule
from mmdet.registry import MODELS

@MODELS.register_module()
class FGD(BaseModule):

    def __init__(
            self,
            channels: int = 256,
            weights: List = (0, 0, 0, 0),
            module_weights: List = (True, True, True, True),
            no_share_params: List = (1, 0.5, 0.5),
    ) -> None:
        super().__init__()
        self.channels = channels
        level_weight = no_share_params[0]
        self.spa_t = no_share_params[1]
        self.chn_t = no_share_params[2]

        self.fg_weight =  weights[0] * level_weight if module_weights[0] else weights[0]
        self.bg_weight = weights[1] * level_weight if module_weights[1] else weights[1]
        self.spa_weight =  weights[2] * level_weight if module_weights[2] else weights[2]
        self.chn_weight = weights[3] * level_weight if module_weights[3] else weights[3]

        self.fg_weight = self.fg_weight * 0.00001
        self.bg_weight = self.bg_weight * 0.00001
        self.spa_weight = self.spa_weight * 0.00001

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

    @staticmethod
    def get_fg_bg_mask(student, batch_data_samples):

        N, c, H, W = student.shape
        img_h, img_w = batch_data_samples[0].img_shape
        fg_mask = torch.zeros_like(student[:, 0])
        bg_mask = torch.ones_like(student[:, 0])

        wmin, wmax, hmin, hmax = [], [], [], []
        for i in range(N):
            gt_bboxes = batch_data_samples[i].gt_instances.bboxes
            new_boxxes = torch.ones_like(gt_bboxes)
            new_boxxes[:, 0] = gt_bboxes[:, 0] / img_w * W
            new_boxxes[:, 2] = gt_bboxes[:, 2] / img_w * W
            new_boxxes[:, 1] = gt_bboxes[:, 1] / img_h * H
            new_boxxes[:, 3] = gt_bboxes[:, 3] / img_h * H

            wmin.append(torch.floor(new_boxxes[:, 0]).int())
            wmax.append(torch.ceil(new_boxxes[:, 2]).int())
            hmin.append(torch.floor(new_boxxes[:, 1]).int())
            hmax.append(torch.ceil(new_boxxes[:, 3]).int())

            area = 1.0 / (hmax[i].view(1, -1) + 1 - hmin[i].view(1, -1)) / (
                    wmax[i].view(1, -1) + 1 - wmin[i].view(1, -1))

            for j in range(len(gt_bboxes)):
                fg_mask[i][hmin[i][j]:hmax[i][j] + 1, wmin[i][j]:wmax[i][j] + 1] = \
                    torch.maximum(fg_mask[i][hmin[i][j]:hmax[i][j] + 1, wmin[i][j]:wmax[i][j] + 1], area[0][j])

            bg_mask[i] = torch.where(fg_mask[i] > 0, 0, 1)
            if torch.sum(bg_mask[i]):
                bg_mask[i] /= torch.sum(bg_mask[i])

        return fg_mask.unsqueeze(1), bg_mask.unsqueeze(1)

    def forward(self, student, teacher, batch_data_samples):
        losses = dict()

        fg_mask, bg_mask = self.get_fg_bg_mask(student, batch_data_samples)

        (spa_attn_s, chn_attn_s, spa_attn_t, chn_attn_t, spa_chn_mask
         ) = self.get_spa_chn_attn_mask(student, teacher, p=1)

        if self.fg_weight > 0:
            loss_fg = self.l2_loss(student, teacher, norm=False,mask=fg_mask)
            losses['loss_fg'] = loss_fg * self.fg_weight

        if self.bg_weight > 0:
            loss_bg = self.l2_loss(student, teacher, norm=False,mask=bg_mask)
            losses['loss_bg'] = loss_bg * self.bg_weight

        if self.spa_weight > 0:
            loss_spatial = self.l2_loss(spa_attn_s, spa_attn_t, norm=False)
            losses['loss_spa'] = self.spa_weight * loss_spatial

        if self.chn_weight > 0:
            loss_channel = self.l2_loss(chn_attn_s, chn_attn_t, norm=False)
            losses['loss_chn'] = self.chn_weight * loss_channel

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