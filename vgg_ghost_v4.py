'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG16_Ghost_bottle': ['bottleneck_64', 'M', 'bottleneck_128', 'M', 'bottleneck_256', 256, 'M', 'bottleneck_512', 512, 'M', 'bottleneck_512', 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

'''
Def _make_divisible 
'''

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

'''
Def depth_conv
'''

def depthwise_conv(inp, oup, kernel_size=3, stride=1, relu=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True) if relu else nn.Sequential(),
    )

'''
Def hard sigmoid
'''

def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

'''
SE unit
'''

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x



'''
Def channel shuffle
'''

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x




class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, ratio=2, dw_size=3, stride=1, padding=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, padding=padding, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, padding=1, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


class GhostBottleneckV02(nn.Module):
    """ Ghost bottleneck w/ optional SE"""
    def __init__(self, in_chs, mid_chs, out_chs, group=16, dw_kernel_size=3,
                 stride=1, padding=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneckV02, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride
        self.group = group

        self.conv1 = nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=1, \
                               padding=1, \
                               groups=self.group, bias=False)

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, padding=padding, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=1,
                                     groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None


        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, padding=1, relu=False)

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                                       padding=padding,
                                       groups=self.group, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        x = channel_shuffle(x, self.group)
        residual = x
        # 1st ghost bottleneck
        x = self.conv1(x)
        x = self.ghost1(x)
        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)
        # 2nd ghost bottleneck
        # x = channel_shuffle(x, self.group)
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x

class GhostModuleV022(GhostModule):
    def __init__(self, inp, oup, kernel_size=3, ratio=2, dw_size=3, stride=1, padding=1, relu=True):
        super().__init__(inp, oup, kernel_size, ratio, dw_size, stride, padding, relu)
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, 1, 1, padding=0, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )


class GhostBottleneckV022(GhostBottleneckV02):
    def __init__(self, in_chs, mid_chs, out_chs, group=4, dw_kernel_size=3,
                 stride=1, padding=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneckV022, self).__init__(in_chs, mid_chs, out_chs, group, dw_kernel_size,
                 stride, padding, act_layer, se_ratio)

        self.conv1 = nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=1, \
                               padding=1, \
                               groups=self.group, bias=False)
        # self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        self.ghost1 = GhostModuleV022(in_chs, mid_chs, padding=padding, relu=True)
        self.ghost2 = GhostModuleV022(mid_chs, out_chs, padding=1, relu=True)

    def forward(self, x):
        x = channel_shuffle(x, self.group)
        residual = x
        # 1st ghost bottleneck
        x = self.conv1(x)
        x = self.ghost1(x)
        # Depth-wise convolution
        # Squeeze-and-excitation
        # x = self.se(x)
        # 2nd ghost bottleneck
        # x = channel_shuffle(x, self.group)
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x


class GhostBottleneckV023(GhostBottleneckV022):
    def __init__(self, in_chs, mid_chs, out_chs, group=4, dw_kernel_size=3,
                 stride=1, padding=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneckV023, self).__init__(in_chs, mid_chs, out_chs, group, dw_kernel_size,
                                                  stride, padding, act_layer, se_ratio)

        self.conv1 = nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=1, \
                               padding=1, \
                               groups=self.group, bias=False)
        self.se = SqueezeExcite(in_chs, se_ratio=0.1)
        self.ghost1 = GhostModuleV022(in_chs, mid_chs, padding=padding, relu=True)
        self.ghost2 = GhostModuleV022(mid_chs, out_chs, padding=1, relu=True)


    def forward(self, x):
        x = channel_shuffle(x, self.group)
        residual = x
        # 1st ghost bottleneck
        # x = self.conv1(x)
        x = self.se(x)
        x = self.ghost1(x)
        # Depth-wise convolution
        # Squeeze-and-excitation
        # x = self.se(x)
        # 2nd ghost bottleneck
        # x = channel_shuffle(x, self.group)
        x = self.ghost2(x)
        #x += x2
        x += self.shortcut(residual)
        return x



class VGG_ghost_v4(nn.Module):
    def __init__(self, vgg_name):
        super(VGG_ghost_v4, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        #width_mult = 1.
        layers = []
        in_channels = 3
        block = GhostBottleneckV023
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif x == 'bottleneck_64':
                output_channel = 64
                hidden_channel = 64
                layers.append(block(in_channels, hidden_channel, output_channel, dw_kernel_size=3, stride=1, group=1))
                in_channels = 64
            elif x == 'bottleneck_128':
                output_channel = 128
                hidden_channel = 128
                layers.append(block(in_channels, hidden_channel, output_channel, dw_kernel_size=3, stride=1, group=16))
                in_channels = 128
            elif x == 'bottleneck_256':
                output_channel = 256
                hidden_channel = 256
                layers.append(block(in_channels, hidden_channel, output_channel, dw_kernel_size=3, stride=1, group=16))
                in_channels = 256
            elif x == 'bottleneck_512':
                output_channel = 512
                hidden_channel = 512
                layers.append(block(in_channels, hidden_channel, output_channel, dw_kernel_size=3, stride=1, group=16))
                in_channels = 512
            else:
                layers += [GhostModule(in_channels, x, kernel_size=3, relu=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

