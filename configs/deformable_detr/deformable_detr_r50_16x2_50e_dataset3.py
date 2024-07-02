# -*- coding: utf-8 -*-
# @Time    : 09/10/2021 16:19
# @Author  : Linhui Dai
# @FileName: deformable_detr_r50_16x2_50e_dota.py.py
# @Software: PyCharm
angle_version = 'dota'
_base_ = [
    '../_base_/datasets/ai_dataset3.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
model = dict(
    type='RotatedDeformableDETR',
    device='cuda',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        # init_cfg=dict(type='Pretrained', checkpoint='/data/2_data_server/cv-01/ao2_a6000/backbone/dota_back.pth')),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[256, 512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(
        type='RotatedDeformableDETRHead',
        num_query=900,
        num_classes=8,
        in_channels=2048,
        sync_cls_avg_factor=True,
        as_two_stage=False,
        # frm_cfgs1=[
        #     dict(in_channels=256, featmap_strides=[8,16,32,64]),
        #     # dict(in_channels=256, featmap_strides=[128, 64, 32, 16])
        # ],
        transformer=dict(
            type='RotatedDeformableDetrTransformer',
            # use_dab=True,
            # high_dim_query_update = True,
            two_stage_num_proposals=900,
            #mixed_selection = False,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', embed_dims=256),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='RotatedDeformableDetrTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                # use_dab=True,
                # high_dim_query_update = True,
                # embed_dims = 256,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            num_points=5
                            )
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range=angle_version,
            norm_factor=None,
            edge_swap=True,
            proj_xy=True,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.35,
            loss_weight=8.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        # loss_bbox=dict(type='RBBoxL1Cost', loss_weight=5.0),
        # loss_bbox=dict('SmoothL1Loss', beta=1.0, loss_weight=1.0),
        reg_decoded_bbox=True,
        # loss_iou=dict(type='GDLoss', loss_type='gwd', loss_weight=5.0),
        loss_iou=dict(type='RotatedIoULoss', loss_weight=5.0),
        # loss_iou=dict(type='KFLoss', loss_weight=5.0),
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='Rotated_HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=8.0),
            reg_cost=dict(type='RBBoxL1Cost', weight=5.0, box_format='xywha'),
            # iou_cost=dict(type='KFIoUCost', iou_mode='iou', weight=5.0)
            iou_cost=dict(type='RotatedIoUCost', iou_mode='iou', weight=5.0)
            # iou_cost=dict(type='GaussianIoUCost', iou_mode='iou', weight=5.0)
        )),
    test_cfg=dict()
)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(800, 800)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
data = dict(
    train=dict(pipeline=train_pipeline, filter_empty_gt=False, version=angle_version),
    val=dict(version=angle_version),
    test=dict(version=angle_version))
optimizer = dict(
    _delete_=True,
    type='AdamW',
    # lr=2e-4,
    lr = 2e-4,
    # lr= 0.001,
    # momentum=0.9,
    weight_decay=0.0001,
    # weight_decay=0.00005,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
evaluation = dict(interval=5, metric='mAP')
# learning policy
# lr_config = dict(policy='step',  warmup_iters=500,warmup_ratio=1.0 / 3,step=[14,28,42,48])
lr_config = dict(policy='step',  warmup_iters=500,warmup_ratio=1.0 / 3,step=[40])
runner = dict(type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(interval=2)
find_unused_parameters = True

# work_dir = '/data/2_data_server/cv-01/ao2_a6000/arirang_dir/topk_test_query300_24_40/'

# work_dir = '/data/2_data_server/cv-01/ao2_a6000/arirang_dir/small_vehicle/dy_topk_test/'

# work_dir = '/data/2_data_server/cv-01/arirang_vehicle/'


# work_dir = '/data/2_data_server/cv-01/ao2_a6000/arirang_dir/aitod/baseline_nonmix_test/'

# work_dir = '/data/2_data_server/cv-01/ao2_a6000/aitod/trainval_base_numquery_900_test/'

# work_dir = '/data/2_data_server/cv-01/ao2_a6000/aitod/trainval_base_numquery_900_test/'

# work_dir = '/data/2_data_server/cv-01/ao2_a6000/aitod/trainval_base_gwd_900_test/'

# work_dir = '/data/2_data_server/cv-01/ao2_a6000/aitod/trainval_base_1200_test/'

# work_dir = '/data/2_data_server/cv-01/ao2_a6000/aitod/trainval_mixed_1500_test/'

# work_dir = '/data/2_data_server/cv-01/ao2_a6000/aitod/trainval_dytopk_1500_test/'


# work_dir = '/data/2_data_server/cv-01/ao2_a6000/aitod/trainval_baseline900_back0_test/'

work_dir = '/data/2_data_server/cv-01/ao2_a6000/aitod/trainval_dytopk900_back0_test/'
