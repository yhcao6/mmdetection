import torch.nn as nn
from mmdet.core import weighted_l1

from ..registry import LOSSES


@LOSSES.register_module
class L1Loss(nn.Module):

    def __init__(self, loss_weight=1.0):
        super(L1Loss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight, *args, **kwargs):
        loss_bbox = self.loss_weight * weighted_l1(pred, target, weight, *args,
                                                   **kwargs)
        return loss_bbox
