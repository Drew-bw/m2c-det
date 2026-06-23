# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional
from torch import Tensor, nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.ops import MultiScaleDeformableAttention
from mmengine.model import BaseModule, ModuleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.registry import MODELS
from .rtdetr_layers import RTDETRTransformerDecoder, RTDETRHybridEncoder
from .utils import get_text_sine_pos_embed, MLP
from .dfine_layers import Gate

class CrossTransformerEncoderLayer(BaseModule):

    def __init__(self,
                 self_attn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
                     dropout=0.0,
                     batch_first=True),
                 cross_attn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     text_dims=512,
                     hidden_dim=256,
                     head_dim=32,
                     num_hyperedge=16,
                     max_iter=10),
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True)),
                 norm_cfg: OptConfigType = dict(type='LN'),
                 init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.self_attn_cfg = self_attn_cfg
        self.cross_attn_cfg = cross_attn_cfg
        if 'batch_first' not in self.self_attn_cfg:
            self.self_attn_cfg['batch_first'] = True
        else:
            assert self.self_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn = DualMultiheadAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)

    def forward(self,
                query: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,
                **kwargs):

        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
            **kwargs)
        query = self.norms[0](query)
        query = self.cross_attn(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos)
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)
        return query, key


class CrossTransformerEncoder(BaseModule):

    def __init__(self,
                 num_layers: int,
                 layer_cfg: ConfigType,
                 num_cp: int = -1,
                 init_cfg: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)
        self.num_layers = num_layers
        self.layer_cfg = layer_cfg
        self.num_cp = num_cp
        assert self.num_cp <= self.num_layers
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            CrossTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims

    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                query_pos: Tensor, key_pos: Tensor, **kwargs):
        for layer in self.layers:
            query, key = layer(
                query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs)
        return query, key

@MODELS.register_module()
class GroundingRTDETRHybridEncoder(RTDETRHybridEncoder):

    def __init__(self,
                 layer_cfg: ConfigType,
                 in_channels: List[int] = [256, 256, 256],
                 use_encoder_idx: List[int] = [2],
                 num_encoder_layers: int = 1,
                 pe_temperature: float = 10000.0,
                 spatial_shapes: Optional[Tuple[Tuple[int, int]]] = None,
                 encode_before_fpn: bool = True,
                 fpn_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super(RTDETRHybridEncoder, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.encode_before_fpn = encode_before_fpn

        # fpn layer
        self.fpn = MODELS.build(fpn_cfg) \
            if fpn_cfg is not None else nn.Identity()

        # encoder transformer
        self.transformer_blocks = nn.ModuleList([
            CrossTransformerEncoder(num_encoder_layers, layer_cfg)
            for _ in range(len(use_encoder_idx))
        ])

        if spatial_shapes is not None:
            for idx in range(len(use_encoder_idx)):
                spatial_shapes = tuple(map(tuple, spatial_shapes))
                position_embedding = self.build_2d_sincos_position_embedding(
                    *spatial_shapes[idx], in_channels[idx])
                self.register_buffer(
                    f'position_embedding_{idx}',
                    position_embedding,
                    persistent=False)

    def encode_forward(self, inputs: Tuple[Tensor], memory_text: Tensor):
        """
        Args:
            inputs (tuple[Tensor]): input features.

        Returns:
            tuple[Tensor]: encoded features.
        """
        assert len(inputs) == len(self.in_channels)
        outs = list(inputs)

        # encoder
        for i, enc_ind in enumerate(self.use_encoder_idx):
            h, w = outs[enc_ind].shape[2:]
            # flatten [B, C, H, W] to [B, HxW, C]
            src_flatten = outs[enc_ind].flatten(2).permute(0, 2,
                                                           1).contiguous()
            pos_embed = getattr(self, f'position_embedding_{enc_ind}', None)
            if pos_embed is None:
                pos_embed = self.build_2d_sincos_position_embedding(
                    w,
                    h,
                    embed_dim=self.in_channels[enc_ind],
                    temperature=self.pe_temperature,
                    device=src_flatten.device)

            bs, n_text, text_dim = memory_text.shape
            pos_text = (
                torch.arange(n_text,
                             device=memory_text.device).float().unsqueeze(
                    0).unsqueeze(-1).repeat(bs, 1, 1))
            key_pos = get_text_sine_pos_embed(pos_text, num_pos_feats=text_dim, exchange_xy=False)

            memory, memory_text = self.transformer_blocks[i](
                src_flatten, key=memory_text, value=memory_text,
                query_pos=pos_embed, key_pos=key_pos)
            outs[enc_ind] = memory.permute(0, 2, 1).contiguous().reshape(
                -1, self.in_channels[enc_ind], h, w)

        return tuple(outs), memory_text

    def forward(self, inputs: Tuple[Tensor], memory_text: Tensor):
        if self.encode_before_fpn:
            inputs, memory_text = self.encode_forward(inputs, memory_text)
            return self.fpn(inputs), memory_text
        else:
            return self.encode_forward(inputs, memory_text)

class GroundingDeformableDetrTransformerDecoderLayer(BaseModule):

    def __init__(self,
                 self_attn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
                     dropout=0.0,
                     batch_first=True),
                 cross_attn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
                     kdim=512,
                     vdim=512,
                     dropout=0.0,
                     batch_first=True),
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True)),
                 norm_cfg: OptConfigType = dict(type='LN'),
                 init_cfg: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)
        self.self_attn_cfg = self_attn_cfg
        self.cross_attn_cfg = cross_attn_cfg
        if 'batch_first' not in self.self_attn_cfg:
            self.self_attn_cfg['batch_first'] = True
        else:
            assert self.self_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        if 'batch_first' not in self.cross_attn_cfg:
            self.cross_attn_cfg['batch_first'] = True
        else:
            assert self.cross_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)

    def forward(self,
                query: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_mask: Tensor = None,
                dn_meta: dict = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                text_feat: Tensor = None,
                text_key_pos: Tensor = None,
                **kwargs):
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
            **kwargs)
        query = self.norms[0](query)
        query = self.cross_attn(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            **kwargs)
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)

        return query

class GroundingRTDETRTransformerDecoder(RTDETRTransformerDecoder):
    """Transformer decoder of RT-DETR."""

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            GroundingDeformableDetrTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        self.ref_point_head = MLP(4, self.embed_dims * 2, self.embed_dims, 2)
        self.norm = nn.Identity()  # without norm

    def forward(self, query: Tensor, value: Tensor, text_feat: Tensor,
                key_padding_mask: Tensor, self_attn_mask: Tensor, dn_meta: Tensor,
                reference_points: Tensor, spatial_shapes: Tensor,
                level_start_index: Tensor, valid_ratios: Tensor,
                reg_branches: nn.ModuleList, cls_branches: nn.ModuleList, **kwargs) -> Tuple[Tensor]:

        assert self.return_intermediate
        assert reg_branches is not None
        assert reference_points.shape[-1] == 4
        # To avoid inverse_sigmoid, remove .sigmoid() in pre_decoder
        # So reference_points is unactivated reference_points
        unact_reference_points = reference_points
        reference_points = unact_reference_points.sigmoid()

        eval_idx = kwargs.pop('eval_idx', -1)
        if eval_idx < 0:
            eval_idx = eval_idx + self.num_layers
            assert eval_idx >= 0

        all_layers_outputs_classes = []
        all_layers_outputs_coords = []

        bs, n_text, _ = text_feat.shape
        pos_text = (
            torch.arange(n_text, device=text_feat.device).float().unsqueeze(
                0).unsqueeze(-1).repeat(bs, 1, 1))
        text_key_pos = get_text_sine_pos_embed(pos_text, num_pos_feats=256, exchange_xy=False)

        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[:, :, None]
            query_pos = self.ref_point_head(reference_points)
            query = layer(
                query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                dn_meta=dn_meta,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                text_feat=text_feat,
                text_key_pos=text_key_pos,
                **kwargs)

            tmp = reg_branches[lid](query)

            if self.training or lid == eval_idx:
                all_layers_outputs_classes.append(cls_branches[lid](query, text_feat))
                all_layers_outputs_coords.append(
                    (tmp + unact_reference_points).sigmoid())

                if not self.training or lid == self.num_layers - 1:
                    break

            unact_reference_points = tmp + unact_reference_points.detach()
            reference_points = unact_reference_points.sigmoid().detach()

        return all_layers_outputs_classes, all_layers_outputs_coords

class DualMultiheadAttention(BaseModule):

    def __init__(
        self,
        embed_dims: int = 256,
        text_dims: int = 512,
        hidden_dim: int = 256,
        head_dim: int = 32,
        num_hyperedge: int = 16,
        max_iter: int = 10
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.text_dims = text_dims
        self.hidden_dim = hidden_dim
        self.num_heads = hidden_dim // head_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.num_hyperedge = num_hyperedge
        self.max_iter = max_iter

        # text
        self.text_k = nn.Linear(text_dims, hidden_dim)
        self.text_v = nn.Linear(text_dims, hidden_dim)

        # coarse
        self.vision_q_coarse = nn.Linear(embed_dims, hidden_dim)
        self.vision_proj_coarse = nn.Linear(hidden_dim, embed_dims)
        self.norm_vision_coarse = nn.LayerNorm(embed_dims)

        # fine
        self.vision_q_fine = nn.Linear(embed_dims, hidden_dim)
        self.hyperedge_q_fine = nn.Linear(embed_dims, hidden_dim)
        self.hyperedge_k_fine = nn.Linear(hidden_dim, hidden_dim)
        self.hyperedge_v_fine = nn.Linear(hidden_dim, hidden_dim)
        self.vision_proj_fine = nn.Linear(hidden_dim, embed_dims)
        self.norm_hyperedge = nn.LayerNorm(hidden_dim)

    def _in_shape(self, x, seq_len, b):
        return x.reshape(b, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _out_shape(self, x, seq_len, b):
        return x.permute(0, 2, 1, 3).reshape(b, seq_len, self.num_heads * self.head_dim)

    def forward(
        self,
        query,
        key,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs
    ):
        if value is None:
            value = key
        b, query_len, c = query.shape
        _, key_len, _ = key.shape

        text_k = self._in_shape(self.text_k(key + key_pos), key_len, b)
        text_v = self._in_shape(self.text_v(value), key_len, b)

        # coarse
        vision_q_coarse = self._in_shape(self.vision_q_coarse(query + query_pos), query_len, b)
        vision_coarse = F.softmax(
            vision_q_coarse @ text_k.transpose(-1, -2) * self.scale, dim=-1) @ text_v
        vision_coarse = self.vision_proj_coarse(self._out_shape(vision_coarse, query_len, b))
        vision_coarse = self.norm_vision_coarse(vision_coarse + query)

        # fine
        hyperedge = fuzzy_c_means_ultra_fast(
            vision_coarse, n_clusters=self.num_hyperedge,
            max_iter=self.max_iter, training=self.training)
        hyperedge_q_fine = self._in_shape(self.hyperedge_q_fine(hyperedge), self.num_hyperedge, b)
        hyperedge_fine = F.softmax(
            hyperedge_q_fine @ text_k.transpose(-1, -2) * self.scale,dim=-1) @ text_v
        hyperedge_fine = self._out_shape(hyperedge_fine, self.num_hyperedge, b)
        hyperedge_fine = self.norm_hyperedge(hyperedge_fine)

        vision_q_fine = self._in_shape(self.vision_q_fine(vision_coarse), query_len, b)
        hyperedge_k_fine = self._in_shape(self.hyperedge_k_fine(hyperedge_fine), self.num_hyperedge, b)
        hyperedge_v_fine = self._in_shape(self.hyperedge_v_fine(hyperedge_fine), self.num_hyperedge, b)
        vision_fine = F.softmax(
            vision_q_fine @ hyperedge_k_fine.transpose(-1, -2) * self.scale, dim=-1) @ hyperedge_v_fine
        vision_fine = self.vision_proj_fine(self._out_shape(vision_fine, query_len, b))
        vision_fine = vision_fine + vision_coarse

        return vision_fine

class TwoMultiheadAttention(BaseModule):

    def __init__(
        self,
        embed_dims: int = 256,
        text_dims: int = 512,
        hidden_dim: int = 256,
        head_dim: int = 32,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.text_dims = text_dims
        self.hidden_dim = hidden_dim
        self.num_heads = hidden_dim // head_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5

        self.m1 = nn.MultiheadAttention(
            embed_dims, num_heads=self.num_heads, kdim=text_dims, vdim=text_dims, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.m2 = nn.MultiheadAttention(
            embed_dims, num_heads=self.num_heads, kdim=text_dims, vdim=text_dims, batch_first=True)

    def forward(
        self,
        query,
        key,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs
    ):
        if value is None:
            value = key
        query = query + self.m1(
            query + query_pos,
            key + key_pos,
            value)[0]
        query = self.norm(query)
        query = query + self.m2(
            query + query_pos,
            key + key_pos,
            value)[0]
        return query

def fuzzy_c_means_ultra_fast(
    x,
    n_clusters,
    m: int = 2,
    epsilon: float = 0.001,
    max_iter: int = 10,
    training: bool = False,
):
    m_inv = 1.0 / (m - 1)

    batch_size, num_points, num_dims = x.shape
    h = w = int(num_points ** 0.5)
    k = int(n_clusters ** 0.5)

    # Initialize memberships
    centers = F.adaptive_avg_pool2d(
        x.transpose(1, 2).reshape(batch_size, num_dims, h, w), (k, k)).flatten(2)
    if not training:
        return centers.transpose(-1, -2)

    prev_memberships = 0.
    for iteration in range(max_iter):
        # Compute distances using batch matrix multiplication for speed
        # x: (batch_size, num_points, num_dims)
        # centers: (batch_size, num_dims, n_clusters)
        x_norm_sq = (x ** 2).sum(dim=2, keepdim=True)  # (batch_size, num_points, 1)
        centers_norm_sq = (centers ** 2).sum(dim=1, keepdim=True)  # (batch_size, 1, n_clusters)
        cross_term = torch.bmm(x, centers)  # (batch_size, num_points, n_clusters)

        distances_sq = x_norm_sq + centers_norm_sq - 2 * cross_term
        distances_sq = torch.clamp(distances_sq, min=1e-10)

        # Efficient membership update
        distances_powered = distances_sq ** (-m_inv)
        memberships = distances_powered / distances_powered.sum(dim=2, keepdim=True)

        # Handle edge cases
        memberships = torch.nan_to_num(memberships, nan=1.0 / n_clusters)

        # Early stopping check
        if iteration > 0 and torch.norm(prev_memberships - memberships) < epsilon:
            break
        prev_memberships = memberships.clone()

        # Update centers using optimized einsum
        weights = memberships ** m
        numerator = torch.einsum('bpc,bpd->bcd', weights, x)  # (batch_size, n_clusters, num_dims)
        denominator = weights.sum(dim=1, keepdim=True).transpose(1, 2)  # (batch_size, n_clusters, 1)
        centers = (numerator / denominator).transpose(1, 2)  # (batch_size, num_dims, n_clusters)

    return centers.transpose(-1, -2).detach()

