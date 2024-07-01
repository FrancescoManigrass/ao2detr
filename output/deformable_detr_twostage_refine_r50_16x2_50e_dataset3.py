angle_version = 'dota'
auto_resume = False
checkpoint_config = dict(interval=2)
data = dict(
    samples_per_gpu=4,
    test=dict(
        ann_file='data/dataset_v3/test/labels/',
        img_prefix='data/dataset_v3/test/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                flip=False,
                img_scale=(
                    800,
                    800,
                ),
                transforms=[
                    dict(type='RResize'),
                    dict(type='RandomFlip'),
                    dict(
                        mean=[
                            123.675,
                            116.28,
                            103.53,
                        ],
                        std=[
                            58.395,
                            57.12,
                            57.375,
                        ],
                        to_rgb=True,
                        type='Normalize'),
                    dict(size_divisor=32, type='Pad'),
                    dict(type='DefaultFormatBundle'),
                    dict(keys=[
                        'img',
                    ], type='Collect'),
                ],
                type='MultiScaleFlipAug'),
        ],
        type='Datasetv3Dataset',
        version='dota'),
    train=dict(
        ann_file='data/dataset_v3/train/labels/',
        filter_empty_gt=False,
        img_prefix='data/dataset_v3/train/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(img_scale=(
                800,
                800,
            ), type='RResize'),
            dict(
                direction=[
                    'horizontal',
                    'vertical',
                    'diagonal',
                ],
                flip_ratio=[
                    0.25,
                    0.25,
                    0.25,
                ],
                type='RRandomFlip',
                version='dota'),
            dict(
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                to_rgb=True,
                type='Normalize'),
            dict(size_divisor=32, type='Pad'),
            dict(type='DefaultFormatBundle'),
            dict(keys=[
                'img',
                'gt_bboxes',
                'gt_labels',
            ], type='Collect'),
        ],
        type='Datasetv3Dataset',
        version='dota'),
    val=dict(
        ann_file='data/dataset_v3/val/labels/',
        img_prefix='data/dataset_v3/val/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                flip=False,
                img_scale=(
                    800,
                    800,
                ),
                transforms=[
                    dict(type='RResize'),
                    dict(type='RandomFlip'),
                    dict(
                        mean=[
                            123.675,
                            116.28,
                            103.53,
                        ],
                        std=[
                            58.395,
                            57.12,
                            57.375,
                        ],
                        to_rgb=True,
                        type='Normalize'),
                    dict(size_divisor=32, type='Pad'),
                    dict(type='DefaultFormatBundle'),
                    dict(keys=[
                        'img',
                    ], type='Collect'),
                ],
                type='MultiScaleFlipAug'),
        ],
        type='Datasetv3Dataset',
        version='dota'),
    workers_per_gpu=2)
data_root = 'data/dataset_v3/'
dataset_type = 'Datasetv3Dataset'
dist_params = dict(backend='nccl')
evaluation = dict(interval=5, metric='mAP')
find_unused_parameters = True
gpu_ids = range(0, 1)
img_norm_cfg = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
load_from = None
log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
    ], interval=50)
log_level = 'INFO'
lr_config = dict(
    policy='step',
    step=[
        40,
    ],
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333)
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=False, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    bbox_head=dict(
        as_two_stage=True,
        bbox_coder=dict(
            angle_range='dota',
            edge_swap=True,
            norm_factor=None,
            proj_xy=True,
            target_means=(
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ),
            target_stds=(
                0.1,
                0.1,
                0.2,
                0.2,
                0.1,
            ),
            type='DeltaXYWHAOBBoxCoder'),
        in_channels=2048,
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.35,
            gamma=2.0,
            loss_weight=8.0,
            type='FocalLoss',
            use_sigmoid=True),
        loss_iou=dict(loss_weight=5.0, type='RotatedIoULoss'),
        num_classes=8,
        num_query=900,
        positional_encoding=dict(
            normalize=True,
            num_feats=128,
            offset=-0.5,
            type='SinePositionalEncoding'),
        reg_decoded_bbox=True,
        sync_cls_avg_factor=True,
        transformer=dict(
            decoder=dict(
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    attn_cfgs=[
                        dict(
                            dropout=0.0,
                            embed_dims=256,
                            num_heads=8,
                            type='MultiheadAttention'),
                        dict(
                            embed_dims=256,
                            num_points=5,
                            type='MultiScaleDeformableAttention'),
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=(
                        'self_attn',
                        'norm',
                        'cross_attn',
                        'norm',
                        'ffn',
                        'norm',
                    ),
                    type='DetrTransformerDecoderLayer'),
                type='RotatedDeformableDetrTransformerDecoder'),
            encoder=dict(
                num_layers=6,
                transformerlayers=dict(
                    attn_cfgs=dict(
                        embed_dims=256, type='MultiScaleDeformableAttention'),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=(
                        'self_attn',
                        'norm',
                        'ffn',
                        'norm',
                    ),
                    type='BaseTransformerLayer'),
                type='DetrTransformerEncoder'),
            two_stage_num_proposals=900,
            type='RotatedDeformableDetrTransformer'),
        type='RotatedDeformableDETRHead',
        with_box_refine=True),
    neck=dict(
        act_cfg=None,
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        kernel_size=1,
        norm_cfg=dict(num_groups=32, type='GN'),
        num_outs=4,
        out_channels=256,
        type='ChannelMapper'),
    test_cfg=dict(),
    train_cfg=dict(
        assigner=dict(
            cls_cost=dict(type='FocalLossCost', weight=8.0),
            iou_cost=dict(iou_mode='iou', type='RotatedIoUCost', weight=5.0),
            reg_cost=dict(box_format='xywha', type='RBBoxL1Cost', weight=5.0),
            type='Rotated_HungarianAssigner')),
    type='RotatedDeformableDETR')
optimizer = dict(
    betas=(
        0.9,
        0.999,
    ),
    lr=0.0002,
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1),
            reference_points=dict(lr_mult=0.1),
            sampling_offsets=dict(lr_mult=0.1))),
    type='AdamW',
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
resume_from = None
runner = dict(max_epochs=50, type='EpochBasedRunner')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        flip=False,
        img_scale=(
            800,
            800,
        ),
        transforms=[
            dict(type='RResize'),
            dict(type='RandomFlip'),
            dict(
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                to_rgb=True,
                type='Normalize'),
            dict(size_divisor=32, type='Pad'),
            dict(type='DefaultFormatBundle'),
            dict(keys=[
                'img',
            ], type='Collect'),
        ],
        type='MultiScaleFlipAug'),
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(img_scale=(
        800,
        800,
    ), type='RResize'),
    dict(
        direction=[
            'horizontal',
            'vertical',
            'diagonal',
        ],
        flip_ratio=[
            0.25,
            0.25,
            0.25,
        ],
        type='RRandomFlip',
        version='dota'),
    dict(
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        to_rgb=True,
        type='Normalize'),
    dict(size_divisor=32, type='Pad'),
    dict(type='DefaultFormatBundle'),
    dict(keys=[
        'img',
        'gt_bboxes',
        'gt_labels',
    ], type='Collect'),
]
work_dir = 'output'
workflow = [
    (
        'train',
        1,
    ),
]
