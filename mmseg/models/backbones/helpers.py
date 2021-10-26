"""Model creation / weight loading / state_dict helpers.

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
from typing import Callable

import torch.nn as nn


def named_apply(fn: Callable,
                module: nn.Module,
                name='',
                depth_first=True,
                include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def adapt_input_conv(in_chans, conv_weight):
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float(
    )  # Some weights are in torch.half, ensure it's float for sum on CPU
    O, i, J, K = conv_weight.shape
    if in_chans == 1:
        if i > 3:
            assert conv_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv_weight = conv_weight.reshape(O, i // 3, 3, J, K)
            conv_weight = conv_weight.sum(dim=2, keepdim=False)
        else:
            conv_weight = conv_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        if i != 3:
            raise NotImplementedError(
                'Weight format not supported by conversion.')
        else:
            # NOTE this strategy should be better than random init, but there
            # could be other combinations of the original RGB input layer
            # weights that'd work better for specific cases.
            repeat = int(math.ceil(in_chans / 3))
            conv_weight = conv_weight.repeat(1, repeat, 1,
                                             1)[:, :in_chans, :, :]
            conv_weight *= (3 / float(in_chans))
    conv_weight = conv_weight.to(conv_type)
    return conv_weight
