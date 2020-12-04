'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
Def hard sigmoid
'''
def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


'''
Def ghost module
'''
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, padding=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride,padding=1, bias=False),
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

'''
Def Ghost Bottleneck
'''
class GhostBottleneck(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, group=4, dw_kernel_size=3,
                 stride=1, padding=1):
        super(GhostBottleneck, self).__init__()
        self.stride = stride
        self.group = group

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, padding=padding, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=1,
                                     groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, padding=1, relu=False)

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                                       padding=padding, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x
        # 1st ghost bottleneck
        x = self.ghost1(x)
        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        # 2nd ghost bottleneck
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x


class VGG_ghost_v3(nn.Module):
    def __init__(self, vgg_name):
        super(VGG_ghost_v3, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        width_mult = 1.
        layers = []
        in_channels = 3
        block = GhostBottleneck
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif x == 'bottleneck_64':
                output_channel = 64
                hidden_channel = 64
                layers.append(block(in_channels, hidden_channel, output_channel, dw_kernel_size=3, stride=1))
                in_channels = 64
            elif x == 'bottleneck_128':
                output_channel = 128
                hidden_channel = 128
                layers.append(block(in_channels, hidden_channel, output_channel, dw_kernel_size=3, stride=1))
                in_channels = 128
            elif x == 'bottleneck_256':
                output_channel = 256
                hidden_channel = 256
                layers.append(block(in_channels, hidden_channel, output_channel, dw_kernel_size=3, stride=1))
                in_channels = 256
            elif x == 'bottleneck_512':
                output_channel = 512
                hidden_channel = 512
                layers.append(block(in_channels, hidden_channel, output_channel, dw_kernel_size=3, stride=1))
                in_channels = 512
            else:
                layers += [GhostModule(in_channels, x, kernel_size=3, relu=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

