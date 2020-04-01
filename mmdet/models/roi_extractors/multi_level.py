from __future__ import division

import torch.nn as nn

from mmdet import ops
from mmdet.core import force_fp32
from ..registry import ROI_EXTRACTORS


@ROI_EXTRACTORS.register_module
class MultiRoIExtractor(nn.Module):
    """Extract RoI features from multi level feature maps.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
    """

    def __init__(self, roi_layer, out_channels, featmap_strides):
        super(MultiRoIExtractor, self).__init__()
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.fp16_enabled = False

    @property
    def num_inputs(self):
        """int: Input feature map levels."""
        return len(self.featmap_strides)

    def init_weights(self):
        pass

    def build_roi_layers(self, layer_cfg, featmap_strides):
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        roi_layers = nn.ModuleList(
            [layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)

        roi_feats = [
            roi_layer(x, rois) for x, roi_layer in zip(feats, self.roi_layers)
        ]
        return roi_feats
