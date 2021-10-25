import torch.nn as nn
from mmcv.cnn import ConvModule, build_norm_layer
from mmcv.cnn.bricks.activation import build_activation_layer
from mmcv.runner import BaseModule

from ..builder import BACKBONES
from ..utils import AvgPool2dSame, DropPath, make_divisible


class PreActBottleneck(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels=None,
                 bottle_ratio=0.25,
                 stride=1,
                 dilation=1,
                 first_dilation=None,
                 groups=1,
                 drop_path_rate=0.1,
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='GroupNorm'),
                 proj_cfg=None,
                 init_cfg=None):
        super(PreActBottleneck, self).__init__(init_cfg=init_cfg)

        first_dilation = first_dilation or dilation
        out_channels = out_channels or in_channels
        mid_channels = make_divisible(out_channels * bottle_ratio)

        if proj_cfg is None:
            self.downsample = ConvModule(
                in_channels,
                out_channels,
                stride=stride,
                dilation=dilation,
                norm_cfg=norm_cfg,
                act_cfg=None)
        else:
            self.downsample = None

        self.norm1 = build_norm_layer(norm_cfg)[1]
        self.activate = build_activation_layer(act_cfg)
        self.conv1 = ConvModule(in_channels, mid_channels, kernel_size=1)
        self.conv2 = ConvModule(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            groups=groups,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            order=('norm', 'act', 'conv'))
        self.conv3 = ConvModule(
            mid_channels,
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            order=('norm', 'act', 'conv'))
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x):
        x_preact = self.norm1(x)

        # shortcut branch
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x_preact)

        # residual branch
        x = self.conv1(x_preact)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.drop_path(x)
        return x + shortcut


class DownsampleConv(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 preact=True,
                 init_cfg=None):
        super(DownsampleConv, self).__init__(init_cfg)
        norm_cfg = None if preact else dict(type='GroupNorm')
        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            norm_cfg=norm_cfg,
            act_cfg=None)

    def forward(self, x):
        return self.conv(x)


class DownsampleAvg(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1,
                 first_dilation=None,
                 preact=True,
                 norm_cfg=None,
                 init_cfg=None):
        super(DownsampleAvg, self).__init__(init_cfg)
        avg_stride = stride if dilation == 1 else 1
        if stride > 1 or dilation > 1:
            avg_pool_fn = AvgPool2dSame if \
                avg_stride == 1 and dilation > 1 else nn.AvgPool2d
            self.pool = avg_pool_fn(
                2, avg_stride, ceil_mode=True, count_include_pad=False)
        else:
            self.pool = nn.Identity()
        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=None)

    def forward(self, x):
        return self.conv(self.pool(x))


@BACKBONES.register_module()
class ResNetV2(BaseModule):

    def __init__(self,
                 layers,
                 channels=(256, 512, 1024, 2048),
                 in_channels=3,
                 global_pool='avg',
                 out_stride=32,
                 width_factor=1,
                 stem_channels=64,
                 stem_type='',
                 avg_down=False,
                 preact=True,
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='GroupNorm'),
                 drop_rate=0.,
                 drop_path_rate=0.,
                 with_cp=False,
                 pretrained=None,
                 init_cfg=None):
        super(ResNetV2, self).__init__(init_cfg)
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.stem_channels = make_divisible(stem_channels * width_factor)

        if isinstance(pretrained, str):
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                ]
