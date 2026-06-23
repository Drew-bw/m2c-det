# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union, Any
from pathlib import Path
from torch import nn
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType
from .dfine import DFINE


@MODELS.register_module()
class CMKDFINE(DFINE):

    def __init__(
        self,
        teacher_config: Union[ConfigType, str, Path],
        teacher_ckpt: Optional[str] = None,
        eval_teacher: bool = True,
        neck_distill: OptConfigType = None,
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
            neck_channels = [256, 256, 256]
            for i, neck_channel in enumerate(neck_channels):
                neck_distill_ = neck_distill.copy()
                neck_distill_['no_share_params'] = no_share_params[i]
                if 'low_order_module' in neck_distill_:
                    neck_distill_['low_order_module']['embed_dims'] = neck_channel
                if 'high_order_module' in neck_distill_:
                    neck_distill_['high_order_module']['embed_dims'] = neck_channel
                self.neck_distill.append(MODELS.build(neck_distill_))

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:

        with torch.no_grad():
            img_feats_tea = self.teacher_model.extract_feat(batch_inputs[:, 3:])

        # ==============student forward===============
        img_feats = self.extract_feat(batch_inputs[:, :3])
        head_inputs_dict = self.forward_transformer(img_feats, batch_data_samples)
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)
        # ==============student forward===============

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

    @property
    def with_neck_distill(self) -> bool:
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck_distill') and self.neck_distill is not None

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

