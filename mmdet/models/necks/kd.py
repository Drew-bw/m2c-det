# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from typing import List
from mmengine.model import BaseModule
from mmdet.registry import MODELS

@MODELS.register_module()
class KD(BaseModule):

    def __init__(
            self,
            channels: int = 256,
            weights: List = (0, ),
            module_weights: List = (True, ),
            no_share_params: List = (1, ),
    ) -> None:
        super().__init__()
        self.channels = channels
        level_weight = no_share_params[0]
        self.loss_kd_weight = weights[0] * level_weight if module_weights[0] else weights[0]
        self.loss_kd_weight = self.loss_kd_weight * 0.00001

    def forward(self, student, teacher, batch_data_samples):
        losses = dict()

        if self.loss_kd_weight > 0:
            loss_kd = self.l2_loss(student, teacher, norm=False)
            losses['loss_kd'] = self.loss_kd_weight * loss_kd

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