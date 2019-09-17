# model settings
norm_cfg = dict(type='BN', requires_grad=False)
model = dict(
    type='RFCN',
    pretrained='open-mmlab://resnet101_caffe',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=1,
        dilations=(1, 1, 1, 2),
        strides=(1, 2, 2, 1),
        norm_cfg=norm_cfg,
        style='caffe'),
    rpn_head=dict(
        type='RPNHead',
        in_channels=1024,
        feat_channels=512,
        anchor_scales=[8, 16, 32],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[16],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    cls_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='PSRoIPool', out_size=7, group_size=7),
        out_channels=21,
        featmap_strides=[16]),
    reg_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='PSRoIPool', out_size=7, group_size=7),
        out_channels=4,
        featmap_strides=[16]),
    bbox_head=dict(
        type='RFCNHead',
        psroipool_size=7,
        in_channels=2048,
        conv_out_channels=1024,
        num_classes=21,
        reg_class_agnostic=True,
        target_means=[.0, .0, .0, .0],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        smoothl1_beta=1 / 9.0,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=6000,
        nms_post=300,
        max_num=300,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=128,
            pos_fraction=0.25,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=6000,
        nms_post=300,
        max_num=300,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(score_thr=0, nms=dict(type='nms', iou_thr=0.3), max_per_img=100)
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
)
# dataset settings
dataset_type = 'FlipVOCDataset'
data_root = 'data/VOCdevkit/'
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'voc0712_trainval.pkl',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'voc07_test.pkl',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'voc07_test.pkl',
        img_prefix=data_root,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0005,
    paramwise_options={
        'bbox_head.conv_new.weight': {
            'lr_mult': 3
        },
        'bbox_head.conv_new.bias': {
            'lr_mult': 3
        }
    })
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=5e-5,
    step=[5])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 6
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/rfcn_r101_caffe_1x_voc0712'
load_from = None
resume_from = None
workflow = [('train', 1)]
