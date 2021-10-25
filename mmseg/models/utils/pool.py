import torch.nn as nn
import torch.nn.functional as F
from mmcv.utils import to_2tuple

from .padding import pad_same


class AvgPool2dSame(nn.AvgPool2d):
    """Tensorflow like 'SAME' wrapper for 2D average pooling."""

    def __init__(self,
                 kernel_size: int,
                 stride=None,
                 padding=0,
                 ceil_mode=False,
                 count_include_pad=True):
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        super(AvgPool2dSame, self).__init__(kernel_size, stride, (0, 0),
                                            ceil_mode, count_include_pad)

    def forward(self, x):
        x = pad_same(x, self.kernel_size, self.stride)
        return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding,
                            self.ceil_mode, self.count_include_pad)
