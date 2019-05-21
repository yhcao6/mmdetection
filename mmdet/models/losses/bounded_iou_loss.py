import torch.nn as nn
from mmdet.core import bounded_iou_loss, delta2bbox

from ..registry import LOSSES


@LOSSES.register_module
class BoundedIoULoss(nn.Module):

    def __init__(self, beta=0.2, eps=1e-3, loss_weight=1.0):
        super(BoundedIoULoss, self).__init__()
        self.beta = beta
        self.eps = eps
        self.loss_weight = loss_weight

    def forward(self, pos_bbox_pred, pos_bbox_target, bbox_weights, pos_rois,
                target_means, target_stds, avg_factor):
        pos_pred_bboxes = delta2bbox(pos_rois, pos_bbox_pred, target_means,
                                     target_stds)
        pos_target_bboxes = delta2bbox(pos_rois, pos_bbox_target, target_means,
                                       target_stds)
        loss_bbox = self.loss_weight * (bbox_weights * bounded_iou_loss(
            pos_pred_bboxes,
            pos_target_bboxes,
            beta=self.beta,
            eps=self.eps,
            reduction='none')).sum()[None] / avg_factor
        return loss_bbox
