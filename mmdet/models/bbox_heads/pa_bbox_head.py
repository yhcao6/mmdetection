# TODO This PAFCRoIHead is implemented specificlly for CONV after
# RoI Pooling, according to the results with SyncBN
import torch
import torch.nn as nn

from mmdet.ops import ConvModule
from ..registry import HEADS
from .convfc_bbox_head import ConvFCBBoxHead


@HEADS.register_module
class PABBoxHead(ConvFCBBoxHead):

    def __init__(self, num_inputs=4, *args, **kwargs):
        super(PABBoxHead, self).__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            *args,
            **kwargs)
        self.num_inputs = num_inputs

        self.fpn_convs = nn.ModuleList()
        for i in range(num_inputs):
            self.fpn_convs.append(
                ConvModule(
                    self.in_channels,
                    self.conv_out_channels,
                    3,
                    padding=1,
                    norm_cfg=self.norm_cfg))

    def init_weights(self):
        super(PABBoxHead, self).init_weights()

    def forward(self, x):
        assert isinstance(x, list)
        assert len(x) == self.num_inputs
        x = [shared_conv(x_) for x_, shared_conv in zip(x, self.fpn_convs)]
        x, _ = torch.stack(x).max(dim=0)
        cls_score, bbox_pred = super(PABBoxHead, self).forward(x)
        return cls_score, bbox_pred
