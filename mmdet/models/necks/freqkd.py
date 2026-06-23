# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from pytorch_wavelets import DWTForward, DWTInverse

@MODELS.register_module()
class FreqKD(BaseModule):

    def __init__(
            self,
            channels: int = 256,
            weights: List = (0, 0, 0, 0),
            module_weights: List = (True, True, True, True),
            dwt_level: List = (1, 2, 3),
            no_share_params: List = (1, 1, 1),
            low_order_module: dict = dict(
                type='MultiheadReduceAttn',
                embed_dims=256,
                num_heads=8,
                spatial_ratio=4),
            high_order_module: dict = dict(
                type='MultiheadHyperGraph',
                embed_dims=256,
                num_heads=8,
                num_hyperedge=16,
                init_cluster_mode='avg'),
    ) -> None:
        super().__init__()
        self.channels = channels
        self.dwt_level = dwt_level
        level_weight = no_share_params[0]
        spatial_ratio = no_share_params[1]
        num_hyperedge = no_share_params[2]

        self.low_freq_weight = weights[0] * level_weight if module_weights[0] else weights[0]
        self.high_freq_weight = weights[1] * level_weight if module_weights[1] else weights[1]
        self.low_order_weight = weights[2] * level_weight if module_weights[2] else weights[2]
        self.high_order_weight = weights[3] * level_weight if module_weights[3] else weights[3]

        self.low_freq_weight = self.low_freq_weight * 0.00001
        # self.high_freq_weight = self.high_freq_weight * 0.00001
        self.low_order_weight = self.low_order_weight * 0.00001
        self.high_order_weight = self.high_order_weight * 0.00001

        self.dwt2d = nn.ModuleList()
        self.idwt2d = nn.ModuleList()
        for dl in self.dwt_level:
            m = DWTForward(J=dl, wave='haar', mode='zero')
            im = DWTInverse(wave='haar', mode='zero')
            # 冻结小波模块（无参数，仅标记为eval模式）
            for param in m.parameters():
                param.requires_grad = False
            for param in im.parameters():
                param.requires_grad = False
            m.eval()
            im.eval()
            self.dwt2d.append(m)
            self.idwt2d.append(im)

        self.lf_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, len(self.dwt_level), kernel_size=1),
            nn.Softmax(dim=1),
            nn.Flatten(1))
        self.hf_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, len(self.dwt_level), kernel_size=1),
            nn.Softmax(dim=1),
            nn.Flatten(1))

        if self.low_order_weight > 0:
            low_order_module['spatial_ratio'] = spatial_ratio
            self.low_order_tea = MODELS.build(low_order_module)
            self.low_order_stu = MODELS.build(low_order_module)

        if self.high_order_weight > 0:
            high_order_module['num_hyperedge'] = num_hyperedge
            self.high_order_tea = MODELS.build(high_order_module)
            self.high_order_stu = MODELS.build(high_order_module)

    def forward(self, student, teacher, batch_data_samples):
        losses = dict()

        low_freq_stu, high_freq_stu, low_freq_tea, high_freq_tea = self.get_lf_hf_feat(
            student, teacher)
        lf_weight = self.lf_proj(teacher + student)
        hf_weight = self.hf_proj(teacher - student)
        low_freq_stu = torch.einsum('bl, blchw->bchw', lf_weight, low_freq_stu)
        low_freq_tea = torch.einsum('bl, blchw->bchw', lf_weight, low_freq_tea)
        high_freq_stu = torch.einsum('bl, blchw->bchw', hf_weight, high_freq_stu)
        high_freq_tea = torch.einsum('bl, blchw->bchw', hf_weight, high_freq_tea)

        if self.low_freq_weight > 0:
            loss_low_freq = self.l2_loss(low_freq_stu, low_freq_tea, norm=False)
            losses['loss_low_freq'] = self.low_freq_weight * loss_low_freq

        if self.high_freq_weight > 0:
            loss_high_freq = self.l2_loss(high_freq_stu, high_freq_tea, norm=True)
            losses['loss_high_freq'] = self.high_freq_weight * loss_high_freq

        if self.low_order_weight > 0:
            low_order_tea = self.low_order_tea(teacher)
            low_order_stu = self.low_order_stu(student)
            loss_low_order = self.l2_loss(low_order_stu, low_order_tea, norm=False)
            losses['loss_low_order'] = self.low_order_weight * loss_low_order

        if self.high_order_weight > 0:
            high_order_tea = self.high_order_tea(teacher)
            high_order_stu = self.high_order_stu(student)
            loss_high_order = self.l2_loss(high_order_stu, high_order_tea, norm=False)
            losses['loss_high_order'] = self.high_order_weight * loss_high_order

        return losses

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def get_lf_hf_feat(self, student, teacher):
        # 统一padding
        pad_h = 1 if student.shape[2] % 2 != 0 else 0
        pad_w = 1 if student.shape[3] % 2 != 0 else 0
        pad = (pad_h, pad_w)
        student_padded = F.pad(student, (0, pad_w, 0, pad_h), mode='constant',
                               value=0) if pad_h + pad_w > 0 else student
        teacher_padded = F.pad(teacher, (0, pad_w, 0, pad_h), mode='constant',
                               value=0) if pad_h + pad_w > 0 else teacher

        lf_feat_stu_lvl, hf_feat_stu_lvl = [], []
        lf_feat_tea_lvl, hf_feat_tea_lvl = [], []
        for lvl in range(len(self.dwt_level)):
            dl = self.dwt_level[lvl]
            min_size = 2 ** dl
            # 尺寸检查：避免分解出错
            if student_padded.shape[2] < min_size or student_padded.shape[3] < min_size:
                lf_feat_stu = student_padded
                hf_feat_stu = torch.zeros_like(student_padded)
                lf_feat_tea = teacher_padded
                hf_feat_tea = torch.zeros_like(teacher_padded)
            else:
                freq_stu = self.dwt2d[lvl](student_padded)
                freq_tea = self.dwt2d[lvl](teacher_padded)

                lr_freq_mask = torch.zeros_like(freq_stu[0])
                hf_freq_mask = [torch.zeros_like(m) for m in freq_stu[1]]

                lf_freq_stu = [freq_stu[0], hf_freq_mask]
                hf_freq_stu = [lr_freq_mask, freq_stu[1]]
                lf_freq_tea = [freq_tea[0], hf_freq_mask]
                hf_freq_tea = [lr_freq_mask, freq_tea[1]]

                lf_feat_stu = self.idwt2d[lvl](lf_freq_stu)
                hf_feat_stu = self.idwt2d[lvl](hf_freq_stu)
                lf_feat_tea = self.idwt2d[lvl](lf_freq_tea)
                hf_feat_tea = self.idwt2d[lvl](hf_freq_tea)

            # 裁剪padding
            lf_feat_stu = self._unpad(lf_feat_stu, pad)
            hf_feat_stu = self._unpad(hf_feat_stu, pad)
            lf_feat_tea = self._unpad(lf_feat_tea, pad)
            hf_feat_tea = self._unpad(hf_feat_tea, pad)

            lf_feat_stu_lvl.append(lf_feat_stu)
            hf_feat_stu_lvl.append(hf_feat_stu)
            lf_feat_tea_lvl.append(lf_feat_tea)
            hf_feat_tea_lvl.append(hf_feat_tea)

        return (torch.stack(lf_feat_stu_lvl, 1),
                torch.stack(hf_feat_stu_lvl, 1),
                torch.stack(lf_feat_tea_lvl, 1),
                torch.stack(hf_feat_tea_lvl, 1))

    def _unpad(self, x, pad):
        pad_h, pad_w = pad
        if pad_h > 0:
            x = x[:, :, :-pad_h, :]
        if pad_w > 0:
            x = x[:, :, :, :-pad_w]
        return x

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

@MODELS.register_module()
class MultiheadHyperGraph(BaseModule):

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_hyperedge: int = 9,
        attn_drop: float = 0.0,
        init_cluster_mode: str = 'avg'
    ):
        super().__init__()
        assert init_cluster_mode in ['avg', 'max', 'random']
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_hyperedge = num_hyperedge
        self.init_cluster_mode = init_cluster_mode

        self.vertex_to_hyperedge = nn.MultiheadAttention(
            embed_dims, num_heads, attn_drop, batch_first=True)
        self.hyperedge_to_vertex = nn.MultiheadAttention(
            embed_dims, num_heads, attn_drop, batch_first=True)
        self.norm_hyperedge = nn.LayerNorm(embed_dims)

    def forward(self, x):
        b, c, h, w = x.shape
        query = x.flatten(2).transpose(1, 2)
        hyperedge = fuzzy_c_means_ultra_fast(
            query, n_clusters=self.num_hyperedge, init_cluster_mode=self.init_cluster_mode)
        hyperedge = hyperedge + self.vertex_to_hyperedge(
            query=hyperedge,
            key=query,
            value=query)[0]
        hyperedge = self.norm_hyperedge(hyperedge)
        query = query + self.hyperedge_to_vertex(
            query=query,
            key=hyperedge,
            value=hyperedge)[0]
        query = query.transpose(1, 2).reshape(b, c, h, w)
        return query

@MODELS.register_module()
class MultiheadReduceAttn(BaseModule):

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        spatial_ratio: int = 4,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self.scale = self.head_dim ** -0.5

        self.spatial_ratio = spatial_ratio
        if spatial_ratio > 1:
            self.sr = nn.Conv2d(embed_dims, embed_dims, kernel_size=spatial_ratio, stride=spatial_ratio)
            self.norm = nn.LayerNorm(embed_dims)

        self.in_proj_q = nn.Linear(embed_dims, embed_dims)
        self.in_proj_k = nn.Linear(embed_dims, embed_dims)
        self.in_proj_v = nn.Linear(embed_dims, embed_dims)
        self.out_proj = nn.Linear(embed_dims, embed_dims)

        nn.init.xavier_uniform_(self.in_proj_q.weight)
        nn.init.xavier_uniform_(self.in_proj_k.weight)
        nn.init.xavier_uniform_(self.in_proj_v.weight)
        nn.init.constant_(self.in_proj_q.bias, 0.)
        nn.init.constant_(self.in_proj_k.bias, 0.)
        nn.init.constant_(self.in_proj_v.bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, x):

        if self.spatial_ratio > 1:
            x_kv = self.sr(x)
            x_kv = x_kv.flatten(2).transpose(1, 2)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x.flatten(2).transpose(1, 2)

        b, c, h, w = x.shape
        query = x.flatten(2).transpose(1, 2)
        key_value = x_kv

        query = self.in_proj_q(query).reshape(
            b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = self.in_proj_k(key_value).reshape(
            b, -1, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        value = self.in_proj_v(key_value).reshape(
            b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_weight = query @ key * self.scale
        attn_weight = F.softmax(attn_weight, dim=-1)
        out = attn_weight @ value
        out = self.out_proj(out.permute(0, 2, 1, 3).flatten(2))
        out = x + out.transpose(-1, -2).reshape(b, c, h, w)
        return out

def fuzzy_c_means_ultra_fast(
    x,
    n_clusters,
    m: int = 2,
    epsilon: float = 0.001,
    max_iter: int = 10,
    init_cluster_mode: str = 'avg',
):
    x = x.detach()
    device = x.device
    m_inv = 1.0 / (m - 1)

    batch_size, num_points, num_dims = x.shape
    # Initialize memberships
    if init_cluster_mode == 'avg':
        h = w = int(num_points ** 0.5)
        k = int(n_clusters ** 0.5)
        centers = F.adaptive_avg_pool2d(
            x.transpose(1, 2).reshape(batch_size, num_dims, h, w), (k, k)).flatten(2)
    elif init_cluster_mode == 'max':
        h = w = int(num_points ** 0.5)
        k = int(n_clusters ** 0.5)
        centers = F.adaptive_max_pool2d(
            x.transpose(1, 2).reshape(batch_size, num_dims, h, w), (k, k)).flatten(2)
    else:
        random_indices = torch.randint(0, num_points, (batch_size, n_clusters), device=device)
        centers = torch.zeros(batch_size, num_dims, n_clusters, device=device)
        for b in range(batch_size):
            centers[b, :, :] = x[b, random_indices[b], :].t()

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
