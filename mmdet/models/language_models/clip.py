# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from clip import clip
from mmengine.model import BaseModule
from mmdet.registry import MODELS

@MODELS.register_module()
class CLIPTextModel(BaseModule):

    def __init__(
            self,
            use_cache: bool = True,
            cache_path: str = '../datasets/text_embedding.pt',
            embed_dims: int = 512,
            name: str = 'ViT-B/32',
            use_align: bool = False,
            **kwargs) -> None:
        super().__init__(**kwargs)
        self.use_cache = use_cache
        if not use_cache:
            self.model, _ = clip.load(name)
            self.processor = clip.tokenize
            self.model = self.model.float()
            self.text_dim = self.model.text_projection.shape[-1]
            self.text_embedding = {}
            for _, param in self.model.named_parameters():
                param.requires_grad = False
        else:
            self.text_embedding = torch.load(cache_path, weights_only=False)
            self.text_dim = self.text_embedding['car'].shape[-1]

        self.use_align = use_align
        if use_align:
            self.align = Residual(SwiGLUFFN(embed_dims))

    def forward(self, batch_data_samples, batch_inputs, **kwargs):
        """Forward function."""
        device = batch_inputs.device
        with torch.no_grad():
            batch_text_feats = []
            for data_samples in batch_data_samples:
                text_feats = []
                for text in data_samples.text:
                    if self.text_embedding.get(text, None) is not None:
                        text_feat = self.text_embedding[text].to(device)
                    else:
                        inp = self.processor(text).to(device)
                        text_feat = self.model.encode_text(inp)
                        text_feat = text_feat / text_feat.norm(p=2, dim=-1, keepdim=True)
                        self.text_embedding[text] = text_feat
                    text_feats.append(text_feat.detach())
                text_feats = torch.cat(text_feats, dim=0)
                batch_text_feats.append(text_feats)
            batch_text_feats = torch.stack(batch_text_feats, dim=0)
        if self.use_align:
            batch_text_feats = self.align(batch_text_feats)

        return batch_text_feats

    def forward_yolov5(self, batch_texts, **kwargs):
        """Forward function."""
        device = next(self.align.parameters()).device
        with torch.no_grad():
            batch_text_feats = []
            for texts in batch_texts:
                text_feats = []
                for text in texts:
                    if self.text_embedding.get(text, None) is not None:
                        text_feat = self.text_embedding[text].to(device)
                    else:
                        inp = self.processor(text).to(device)
                        text_feat = self.model.encode_text(inp)
                        text_feat = text_feat / text_feat.norm(p=2, dim=-1, keepdim=True)
                        self.text_embedding[text] = text_feat
                    text_feats.append(text_feat.detach())
                text_feats = torch.cat(text_feats, dim=0)
                batch_text_feats.append(text_feats)
            batch_text_feats = torch.stack(batch_text_feats, dim=0)
        if self.use_align:
            batch_text_feats = self.align(batch_text_feats)

        return batch_text_feats

class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        embed_dims: int = 256,
        ratio: int = 4,
    ) -> None:
        super().__init__()
        hidden_dim = embed_dims * ratio
        self.w12 = nn.Linear(embed_dims, hidden_dim)
        self.w3 = nn.Linear(hidden_dim // 2, embed_dims)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class Residual(nn.Module):
    def __init__(self, m) -> None:
        super().__init__()
        self.m = m
        nn.init.zeros_(self.m.w3.bias)
        # For models with l scale, please change the initialization to
        # nn.init.constant_(self.m.w3.weight, 1e-6)
        nn.init.zeros_(self.m.w3.weight)

    def forward(self, x):
        return x + self.m(x)
