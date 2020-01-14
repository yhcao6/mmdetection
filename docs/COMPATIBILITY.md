# Compatibility with Common Libraries

## Compatibility with MMDetection 1.0

### Coordinate System

MMDetection 1.1 addresses some legacy issues following Detectron2. As a result, their models are not compatible:
running inference with the same model weights will produce different results in the two code bases.
Thus, MMDetection 1.1 re-benchmark all the models and provid their links and logs in the model zoo.

The major differences regarding inference are all about the new coordinate system firstly adopted by Detectron2.
This system treats the center of the most left-top pixel as (0, 0) rather than the left-top corner of that pixel.
Thus, the system interprets the coordinates in COCO bounding box and segmentation annotations as coordinates in range `[0, width]` or `[0, height]`.
This affects all the computation related to the bbox and pixel selection,
which might be more natural.

- The height and width of a box with corners (x1, y1) and (x2, y2) is now computed as `width = x2 - x1` and `height = y2 - y1`.
In MMDetection 1.0 and previous version, a "+ 1" was added both height and width.
This modification are in two folds:

  1. Encoding/decoding in bounding box regression.
  2. Iou calculation. This affects the matching process between ground truth and bounding box and the NMS process. The effect here is very negligible, though, the modification makes the iou calculation more consistent and accurate intuitively.

- The anchors are center-aligned to feature grid points and in float type.
In MMDetection 1.0 and previous version, the anchors are in int type and not center-aligned.
This affects the anchor generation in RPN and all the anchor-based methods.

- ROIAlign is implemented differently. The new implementation is [adopted from Detectron2](https://github.com/facebookresearch/detectron2/tree/master/detectron2/layers/csrc/ROIAlign).

  1. All the ROIs are shifted by half a pixel compared to MMDetection 1.0 in order to better allign the image coordinate system.
     To enable the old behavior, set `ROIAlign(aligned=False)`instead of
     `ROIAlign(aligned=True)` (the default).

  2. The ROIs are not required to have a minimum size of 1.
     This will lead to tiny differences in the output, but should be negligible.

- Mask crop and paste function is different.

  1. We use the new RoIAlign to crop mask target. In MMDetection 1.0, the bounding box is quantilized before it is used to crop mask target, and the crop process is implemented by numpy. In new implementation, the bounding box for crop is not quantilized and sent to RoIAlign. This implementation accelerates the training speed by a large margin (~0.1s per iter, ~2 hour when training Mask R50 for 1x schedule)and should be more accurate.

  2. In MMDetection 1.1, the "paste_mask" function is different and should be more accurate than those in previous versions. This change follows the modification in [Detectron2](https://github.com/facebookresearch/detectron2/blob/master/detectron2/structures/masks.py) and can improve mask AP on COCO by ~0.5% absolute.

### Some Conventions

- MMDetection 1.1 adopts the same convention in the order of class labels as in [Detectron2](https://github.com/facebookresearch/detectron2) .
This effect all the classification layers of the model to have a different ordering of class labels.

  - In MMDetection 1.1, label "K" means background, and labels [0, K-1] correspond to the K = num_categories object categories.

  - In MMDetection 1.0 and previous version, label "0" means background, and labels [1, K] correspond to the K categories.

- Low quality matching in RCNN is not used. In MMDetection 1.0 and previous versions, the `max_iou_assigner` will match low quality boxes for each ground truth box in both rpn and rcnn training. We observe this sometimes does not assign the most perfect GT box to some bounding boxes,
thus MMDetection 1.1 do not allow low quality matching by default in rcnn training in the new system. This slightly improve the box AP (~0.1% absolute).

- Seperate scale factors for width and height. In MMDetection 1.0 and previous versions, the scale factor is a single float in mode `keep_ratio=True`. This is slightly inaccurate because the scale factors for width and height have slight difference. MMDetection 1.1 adopts separate scale factors for width and height, the improvment on AP is negligible, though.

### Training Hyperparameters

There are some other differences in training as well.
These modification does not affect
model-level compatibility but could slight improve the performance. The major ones are:

- We change the number of proposals after nms  from 2000 to 1000 by setting `nms_post=1000` and `max_num=1000`.
This slightly improves both mask AP and bbox AP by ~0.2% absolute.

- For simplicity,  change the default loss in bounding box regression to L1 loss, instead of smooth L1 loss. This leads to an overall improvement in box AP (~0.4% absolute).

## Compatibility with Detectron2

There are some slight difference between MMDetection 1.1 and Detectron2 due to legacy and performance issues:

- MMDetection 1.1 uses OpenCV while Detectron2 uses PIL to transform images. The resized output for the same image is slightly different due to different resize implementation of OpenCV and PIL. Using PIL for resizing image in training will cause ~0.2 mAP drop in MMDetection.

- MMDetection 1.1 uses the original NMS implementation while Detectron2 uses the NMS provided by torchvision. These two NMS produce slightly different outputs (~3 bboxes are different over 200 bboxes). Use NMS from torchvision in training also cause ~0.4 mAP drop in our MMDetection and currently the reason is unknown. So for now MMDetection does not use the NMS from torchvision.

- MMDetection 1.1 treats the mask as an image in transformation while Detectron2 transforms the mask's polygon. If the mask is flipped in polygon format, the output will not be the exact mirror of the un-flipped mask (several pixels will be different). The influence of this difference to the training process is negligible, so MMDetection 1.1 still treats the mask as an image during transformation.

As a result, running inference with the same model weights will produce slightly different results in the two code bases,
but the gap between mAP should be less than 0.1.
