_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

dataset_type = 'VehicleDataset'

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    test=dict(
        type=dataset_type,
        ann_file='data/vehicle_val.json',
        img_prefix='data/snapshots/20200929'))
evaluation = dict(interval=1, metric='bbox')
