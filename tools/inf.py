from mmdet.apis import init_detector, inference_detector, show_result

config_file = 'configs/cityscapes/mask_rcnn_r50_fpn_1x_cityscapes.py'
checkpoint_file = 'models/mask_rcnn_r50_fpn_1x_city_20190727-9b3c56a5.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'data/cityscapes/cityscapes/val/frankfurt_000000_015676_leftImg8bit.png'
# or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# or save the visualization results to image files
show_result(img, result, model.CLASSES, show=False, out_file='img/result.jpg')
