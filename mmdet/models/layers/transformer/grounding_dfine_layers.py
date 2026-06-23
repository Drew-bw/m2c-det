# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple
import torch.nn.functional as F
from torch import Tensor, nn
from copy import deepcopy
from mmcv.ops import MultiScaleDeformableAttention
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmengine.model import BaseModule, ModuleList
from mmdet.utils import OptConfigType
from .dfine_layers import (DFINETransformerDecoder, distance2bbox, Gate, LQE, Integral,
                           MultiNumPointsMultiScaleDeformableAttention)
from .utils import MLP

class GroundingDFINETransformerDecoderLayer(BaseModule):
    """Decoder layer of D-FINE."""

    def __init__(self,
                 self_attn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
                     dropout=0.0,
                     batch_first=True),
                 cross_attn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
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
        num_points = self.cross_attn_cfg.get('num_points', None)
        if num_points is None or isinstance(num_points, int):
            self.cross_attn = MultiScaleDeformableAttention(
                **self.cross_attn_cfg)
        else:
            self.cross_attn = MultiNumPointsMultiScaleDeformableAttention(
                **self.cross_attn_cfg)
        self.cross_attn.value_proj = nn.Identity()
        self.cross_attn.output_proj = nn.Identity()

        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)
        self.gateway = Gate(self.embed_dims)

    def forward(self,
                query: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                text_feat: Tensor = None,
                **kwargs) -> Tensor:
        """
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """

        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
            **kwargs)
        query = self.norms[0](query)
        kwargs.pop('identity', None)
        query_ = self.cross_attn(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            text_feat=text_feat,
            identity=0,
            **kwargs)
        query = self.gateway(query, query_)
        query = self.norms[1](query)

        query = self.ffn(query)
        query = query.clamp(min=-65504, max=65504)
        query = self.norms[2](query)

        return query

class GroundingDFINETransformerDecoder(DFINETransformerDecoder):

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        num_wide_layers = self.num_layers - self.eval_idx - 1
        self.layers = ModuleList([
            GroundingDFINETransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers - num_wide_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims

        if num_wide_layers > 0:
            wide_layer_cfg = self.layer_cfg.deepcopy()

            scaled_dim = int(round(self.layer_scale * self.embed_dims))
            if scaled_dim != self.embed_dims:
                for key in {'self_attn_cfg', 'cross_attn_cfg', 'ffn_cfg'}:
                    if key not in wide_layer_cfg:
                        import inspect
                        parameters = inspect.signature(
                            GroundingDFINETransformerDecoderLayer.__init__).parameters
                        wide_layer_cfg[key] = deepcopy(parameters[key].default)
                    wide_layer_cfg[key]['embed_dims'] = scaled_dim

            self.layers.extend([
                GroundingDFINETransformerDecoderLayer(**wide_layer_cfg)
                for _ in range(num_wide_layers)
            ])
        self.scaled_dim = self.layers[-1].embed_dims

        if self.remove_cross_attn_value_proj_and_output_proj:
            for layer in self.layers:
                layer.cross_attn.value_proj = nn.Identity()
                layer.cross_attn.output_proj = nn.Identity()

        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')

        self.ref_point_head = MLP(
            4,
            self.ref_hidden_dim or self.embed_dims * 2,
            self.embed_dims,
            self.ref_num_layers,
            act_cfg=self.ref_act_cfg)

        self.integral = Integral(self.reg_max, self.reg_scale)
        self.lqe_layers = ModuleList([
            LQE(4, 64, 2, self.reg_max, act_cfg=self.lqe_act_cfg)
            for _ in range(self.num_layers)
        ])

    def forward(self, query: Tensor, value: Tensor, text_feat: Tensor,
                key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                cls_branches: nn.ModuleList, **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.
            cls_branches: (obj:`nn.ModuleList`): Used for classification
                results.

        Returns:
            tuple[Tensor]: Output queries and references of Transformer
                decoder

            - query (Tensor): Output embeddings of the last decoder, has
              shape (num_queries, bs, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, num_queries, bs,
              embed_dims).
            - reference_points (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 4)  when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (num_decoder_layers, bs, num_queries, 4). The
              coordinates are arranged as (cx, cy, w, h)
        """
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
        assert eval_idx == self.eval_idx

        all_layers_outputs_classes = []
        all_layers_outputs_coords = []
        all_layers_outputs_corners = []

        query_detach = 0
        pred_corners_undetach = 0

        assert len(cls_branches) == self.num_layers + 1
        assert len(reg_branches) == self.num_layers + 2
        pre_bbox_head = reg_branches[-1]

        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[:, :, None]
            query_pos = self.ref_point_head(reference_points)
            query_pos = query_pos.clamp(min=-10, max=10)

            # Adjust scale if needed for detachable wider layers
            if lid > self.eval_idx and self.scaled_dim != self.embed_dims:
                if self.scaled_dim != query_pos.size(-1):
                    query_pos = F.interpolate(query_pos, size=self.scaled_dim)
                if self.scaled_dim != query.size(-1):
                    query = F.interpolate(query, size=self.scaled_dim)
                    query_detach = query.detach()
                if self.scaled_dim != value.size(-1):
                    value = F.interpolate(value, size=self.scaled_dim)

            query = layer(
                query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                text_feat=text_feat,
                **kwargs)

            if lid == 0:
                reference_points_initial = \
                    (pre_bbox_head(query) + unact_reference_points).sigmoid()
                reference_points_initial_detach = \
                    reference_points_initial.detach()

                if self.training:
                    all_layers_outputs_classes.append(cls_branches[0](query, text_feat))
                    all_layers_outputs_coords.append(reference_points_initial)

            # Refine bounding box corners using FDR,
            # integrating previous layer's corrections
            pred_corners = reg_branches[lid](
                query + query_detach) + pred_corners_undetach
            new_reference_points = distance2bbox(
                reference_points_initial_detach,
                self.integral(pred_corners),
                self.reg_scale,
                clamp_wh=True)

            if self.training or lid == eval_idx:
                # Lqe does not affect the performance here.
                scores = self.lqe_layers[lid](
                    cls_branches[lid](query, text_feat), pred_corners)
                all_layers_outputs_classes.append(scores)
                all_layers_outputs_coords.append(new_reference_points)
                all_layers_outputs_corners.append(pred_corners)

                if not self.training or lid == self.num_layers - 1:
                    break

            query_detach = query.detach()
            pred_corners_undetach = pred_corners
            reference_points = new_reference_points.detach()

        if self.training:
            all_layers_outputs_coords = (all_layers_outputs_coords,
                                         all_layers_outputs_corners)

        return all_layers_outputs_classes, all_layers_outputs_coords