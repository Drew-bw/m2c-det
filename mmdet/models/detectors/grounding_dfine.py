# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch.nn as nn
from ..layers import (DFINECdnQueryGenerator, GroundingDFINETransformerDecoder,
                      GroundingRTDETRHybridEncoder)
from ..layers.transformer.dfine_layers import (
    LQE, Gate, MultiNumPointsMultiScaleDeformableAttention)
from ..language_models import CLIPTextModel
from .grounding_rtdetr import GroundingRTDETR
from mmdet.registry import MODELS
from mmdet.utils import OptConfigType

@MODELS.register_module()
class GroundingDFINE(GroundingRTDETR):

    def __init__(self, *args, dn_cfg: OptConfigType = None, **kwargs) -> None:
        super().__init__(*args, dn_cfg=dn_cfg, **kwargs)
        self.dn_query_generator = DFINECdnQueryGenerator(**dn_cfg)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.encoder = GroundingRTDETRHybridEncoder(**self.encoder)
        self.decoder = GroundingDFINETransformerDecoder(**self.decoder)
        self.language_model = CLIPTextModel(**self.language_model)
        self.embed_dims = self.decoder.embed_dims
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        for m in self.modules():
            if isinstance(m, (Gate, MultiNumPointsMultiScaleDeformableAttention)):
                m.init_weights()
            elif isinstance(m, LQE):
                for layer in m.reg_conf.layers[:-1]:
                    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                m.init_weights()