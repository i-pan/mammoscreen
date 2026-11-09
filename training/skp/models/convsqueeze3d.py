import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import Conv2dSame
from timm.layers.padding import get_same_padding
from typing import List, Optional, Tuple


def pad_same(
    x,
    kernel_size: List[int],
    stride: List[int],
    dilation: List[int] = (1, 1, 1),
    value: float = 0,
):
    it, ih, iw = x.size()[-3:]
    # do not pad T dim, allow it to collapse to 1
    pad_t = 0
    pad_h = get_same_padding(ih, kernel_size[1], stride[1], dilation[1])
    pad_w = get_same_padding(iw, kernel_size[2], stride[2], dilation[2])
    padding = (
        pad_w // 2,
        pad_w - pad_w // 2,
        pad_h // 2,
        pad_h - pad_h // 2,
        pad_t // 2,
        pad_t - pad_t // 2,
    )
    x = F.pad(
        x,
        padding,
        value=value,
    )
    return x


def conv3d_same(
    x,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Tuple[int, int, int] = (1, 1, 1),
    padding: Tuple[int, int, int] = (0, 0, 0),
    dilation: Tuple[int, int, int] = (1, 1, 1),
    groups: int = 1,
):
    x = pad_same(x, weight.shape[-3:], stride, dilation)
    return F.conv3d(x, weight, bias, stride, (0, 0, 0), dilation, groups)


class Conv3dSame(nn.Conv3d):
    """Tensorflow like 'SAME' convolution wrapper for 2D convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            0,
            dilation,
            groups,
            bias,
        )

    def forward(self, x):
        return conv3d_same(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class ConvSqueeze3d(nn.Conv3d):
    """
    3D convolution which squeezes the depth dimension to 1, essentially producing a 3D output.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups=1,
        bias=True,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        assert x.size(2) == self.kernel_size[0]
        x = super().forward(x)
        assert x.size(2) == 1
        return x[:, :, 0]

    def load_weights_from_conv2d(self, conv2d):
        state_dict = conv2d.state_dict()
        for k, v in state_dict.items():
            if k == "weight":
                v = v.unsqueeze(2)
                v = v.repeat(1, 1, self.kernel_size[0], 1, 1)
                v = v / self.kernel_size[0]
                state_dict[k] = v
        self.load_state_dict(state_dict)

    @classmethod
    def from_conv2d(cls, conv2d, depth):
        in_channels = conv2d.in_channels
        out_channels = conv2d.out_channels
        kernel_size = (depth, conv2d.kernel_size[0], conv2d.kernel_size[1])
        stride = (1, conv2d.stride[0], conv2d.stride[1])
        padding = (0, conv2d.padding[0], conv2d.padding[1])
        groups = conv2d.groups
        bias = conv2d.bias is not None
        conv_squeeze = cls(
            in_channels, out_channels, kernel_size, stride, padding, groups, bias
        )
        conv_squeeze.load_weights_from_conv2d(conv2d)
        return conv_squeeze


class ConvSqueeze3dSame(ConvSqueeze3d):
    def forward(self, x):
        assert x.size(2) == self.kernel_size[0]
        x = conv3d_same(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        assert x.size(2) == 1
        return x[:, :, 0]


def replace_first_conv2d_layer_with_convsqueeze3d(model, depth):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if isinstance(module, Conv2dSame):
                conv_layer = ConvSqueeze3dSame
            else:
                conv_layer = ConvSqueeze3d
            squeezeconv = conv_layer.from_conv2d(module, depth=depth)
            parent_module_name = name.rsplit(".", 1)[0] if "." in name else None
            if parent_module_name:
                parent_module = model.get_submodule(parent_module_name)
                setattr(parent_module, name.split(".")[-1], squeezeconv)
            else:
                setattr(model, name, squeezeconv)
            break
