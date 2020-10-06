_base_ = '../htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_c' \
         'oco.py'

dataset_type = 'VehicleDataset'

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    test=dict(
        type=dataset_type,
        ann_file='data/vehicle_val.json',
        img_prefix='data/snapshots/20200929'))
evaluation = dict(interval=1, metric='bbox')
