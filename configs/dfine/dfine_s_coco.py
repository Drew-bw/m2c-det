_base_ = ['../_base_/default_runtime.py']
pretrained = 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'  # noqa
num_queries = 300
num_classes = 80
base_size_repeat = 20
base_dim = 256
num_points = [3, 6, 3]
reg_max = 32
reg_scale = 4
layer_scale = 1.0
eval_idx = -1

data_preprocessor_stage1 = dict(
    type='DetDataPreprocessor',
    batch_augments=[
        dict(
            type='BatchSyncRandomResize',
            interval=1,
            interpolations='nearest',
            random_sizes=[480, 512, 544, 576, 608] +
            [640] * base_size_repeat + [672, 704, 736, 768, 800])
    ],
    mean=[0, 0, 0],
    std=[255, 255, 255],
    bgr_to_rgb=True,
    pad_size_divisor=1)
data_preprocessor_stage2 = dict(
    type='DetDataPreprocessor',
    mean=[0, 0, 0],
    std=[255, 255, 255],
    bgr_to_rgb=True,
    pad_size_divisor=1)

model = dict(
    type='DFINE',
    num_queries=num_queries,
    with_box_refine=True,
    as_two_stage=True,
    eval_idx=eval_idx,
    data_preprocessor=data_preprocessor_stage1,
    backbone=dict(
        type='HGNetV2',
        name='B0',
        return_idx=[1, 2, 3],
        freeze_at=-1,
        freeze_norm=False,
        use_lab=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='ChannelMapper',
        in_channels=[256, 512, 1024],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='BN', requires_grad=True),
        num_outs=3),
    encoder=dict(
        use_encoder_idx=[-1],
        num_encoder_layers=1,
        in_channels=[256, 256, 256],
        fpn_cfg=dict(
            type='DFINEFPN',
            num_csp_blocks=1,
            in_channels=[256, 256, 256],
            out_channels=256,
            expansion=0.5,
            norm_cfg=dict(type='BN', requires_grad=True)),
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256,
                num_heads=8,
                dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=1024,
                ffn_drop=0.0,
                act_cfg=dict(type='GELU')))),
    decoder=dict(
        reg_max=reg_max,
        reg_scale=reg_scale,
        layer_scale=layer_scale,
        eval_idx=eval_idx,
        num_layers=3,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256,
                num_heads=8,
                dropout=0.0),
            cross_attn_cfg=dict(
                embed_dims=256,
                num_points=num_points,
                num_levels=3,
                dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=1024,
                ffn_drop=0.0)),
        post_norm_cfg=None),
    bbox_head=dict(
        type='DFINEHead',
        num_classes=num_classes,
        reg_max=reg_max,
        reg_scale=reg_scale,
        layer_scale=layer_scale,
        eval_idx=eval_idx,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='RTDETRVarifocalLoss',  # FocalLoss in DINO
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        loss_ld=dict(
            type='KnowledgeDistillationKLDivLoss',
            T=5, reduction='none',
            loss_weight=1.5)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=1000))

# dataset settings
dataset_type = 'CocoDataset'
data_root = '../datasets/coco/'
backend_args = None
train_pipeline_stage1 = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomApply',
        transforms=dict(
            type='PhotoMetricDistortion',
            hue_delta=12.75,
            swap_channel=False,
            clip_val=255,
            force_float32=False),
        prob=0.5),
    dict(type='Expand', mean=[0, 0, 0]),
    dict(
        type='RandomApply',
        transforms=dict(
            type='MinIoURandomCrop', cover_all_box=False, trials=40),
        prob=0.8),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='Resize', scale=(640, 640), keep_ratio=False),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction'))
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='Resize', scale=(640, 640), keep_ratio=False),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(640, 640), keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    # batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline_stage1,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

# numpy < 1.24.0
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

# set all norm layers in backbone to lr_mult=0.5 and decay_mult=0.0
# set all other layers in backbone to lr_mult=0.5
num_blocks_list = (1, 1, 2, 1)
backbone_norm_multi = dict(lr_mult=0.5, decay_mult=0)  # NOTE decay_mult=0 ?
custom_keys = {
    'backbone': dict(lr_mult=0.5), 'in_proj_bias': dict(decay_mult=0)}
custom_keys.update({
    f'backbone.stem.{name}.bn': backbone_norm_multi
    for name in ['stem1', 'stem2a', 'stem2b', 'stem3', 'stem4']
})
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.layers.{lid}.bn':
    backbone_norm_multi
    for stage_id, num_blocks in enumerate((1, 1))
    for block_id in range(num_blocks)
    for lid in range(3)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.layers.{lid}.conv{cid}.bn':
    backbone_norm_multi
    for stage_id, num_blocks in enumerate(num_blocks_list[2:], start=2)
    for block_id in range(num_blocks)
    for lid in range(3)
    for cid in (1, 2)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.aggregation.{lid}.bn':
    backbone_norm_multi
    for stage_id, num_blocks in enumerate(num_blocks_list)
    for block_id in range(num_blocks)
    for lid in range(2)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.downsample.bn': backbone_norm_multi
    for stage_id in range(1, 4)
})

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0004, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys=custom_keys,
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True
    )
)

# learning policy
max_epochs = 12
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.002, by_epoch=False, begin=0, end=500),
]

stage2_num_epochs = 1
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49),
    dict(
        type='DataPreprocessorSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_data_preprocessor=data_preprocessor_stage2),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

default_hooks = dict(
    visualization=dict(type='DetVisualizationHook'),
    logger=dict(type='LoggerHook', interval=500),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1),
)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (16 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(enable=False, base_batch_size=64)

find_unused_parameters = False
randomness = dict(seed=0, deterministic=False)
