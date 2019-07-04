import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import ConvModule


@NECKS.register_module
class NASFPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None,
                 num_laterals=5,
                 num_fpn_convs=7):
        super(NASFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.fp16_enabled = False

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.num_laterals = num_laterals
        self.num_fpn_convs = num_fpn_convs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(start_level, len(in_channels)):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            self.lateral_convs.append(l_conv)
        for i in range(len(in_channels), self.num_laterals + 1):
            l_conv = ConvModule(
                in_channels[-1],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            self.lateral_convs.append(l_conv)

        for i in range(num_fpn_convs):
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            self.fpn_convs.append(fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        _, c3, c4, c5 = inputs
        c6 = F.max_pool2d(c5, 2, 2)
        c7 = F.max_pool2d(c5, 4, 4)
        cs = [c3, c4, c5, c6, c7]
        ps = []
        for i in range(len(cs)):
            ps.append(self.lateral_convs[i](cs[i]))
        p3, p4, p5, p6, p7 = ps

        outs = []
        p6_to_p4 = F.interpolate(p6, size=p4.size()[-2:])
        p4_1 = self.merge_gp(p6_to_p4, p4)
        p4_1 = self.fpn_convs[0](p4_1)
        p4_2 = self.merge_sum(p4, p4_1)
        p4_2 = self.fpn_convs[1](p4_2)
        p4_2_to_p3 = F.interpolate(p4_2, size=p3.size()[-2:])
        p3_3 = self.merge_sum(p4_2_to_p3, p3)
        p3_3 = self.fpn_convs[2](p3_3)
        outs.append(p3_3)

        p3_3_to_p4 = F.max_pool2d(p3_3, 2, 2)
        p4_4 = self.merge_sum(p4_2, p3_3_to_p4)
        p4_4 = self.fpn_convs[3](p4_4)
        outs.append(p4_4)

        p4_4_to_p5 = F.max_pool2d(p4_4, 2, 2)
        p3_3_to_p5 = F.max_pool2d(p3_3, 4, 4)
        gp_p4_4_p3_3 = self.merge_gp(p4_4_to_p5, p3_3_to_p5)
        p5_5 = self.merge_sum(gp_p4_4_p3_3, p5)
        p5_5 = self.fpn_convs[4](p5_5)
        outs.append(p5_5)

        p4_2_to_p7 = F.max_pool2d(p4_2, 8, 8)
        p5_5_to_p7 = F.max_pool2d(p5_5, 4, 4)
        gp_p5_5_p4_2 = self.merge_gp(p5_5_to_p7, p4_2_to_p7)
        p7_6 = self.merge_sum(gp_p5_5_p4_2, p7)
        p7_6 = self.fpn_convs[5](p7_6)

        p7_6_to_p6 = F.interpolate(p7_6, size=p6.size()[-2:])
        p5_5_to_p6 = F.max_pool2d(p5_5, 2, 2)
        p6_7 = self.merge_gp(p7_6_to_p6, p5_5_to_p6)
        p6_7 = self.fpn_convs[6](p6_7)
        outs.append(p6_7)
        outs.append(p7_6)
        return tuple(outs)

    def merge_sum(self, f1, f2):
        return f1 + f2

    def merge_gp(self, f1, f2):
        gp = F.max_pool2d(f1, f1.shape[-2:]).sigmoid()
        return f1 + gp * f2

