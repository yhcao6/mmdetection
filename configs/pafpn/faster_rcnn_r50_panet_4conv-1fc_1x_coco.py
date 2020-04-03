_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

model = dict(
    type='FasterRCNN',
    neck=dict(
        type='PAFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU')),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='MultiRoIExtractor',
            roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='PABBoxHead', conv_out_channels=256, norm_cfg=norm_cfg)))
