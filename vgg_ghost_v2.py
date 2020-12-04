'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
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
Def depth_conv
'''

def depthwise_conv(inp, oup, kernel_size=3, stride=1, relu=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True) if relu else nn.Sequential(),
    )

'''
Def ghost module
'''
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
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
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]
        '''
        # Squeeze-and-Excite
        SELayer(hidden_dim) if use_se else nn.Sequential(),
        '''
        self.conv = nn.Sequential(
            # pw
            GhostModule(inp, hidden_dim, kernel_size=1, relu=True),
            # dw
            depthwise_conv(hidden_dim, hidden_dim, kernel_size, stride, relu=False) if stride==2 else nn.Sequential(),
            # pw-linear
            GhostModule(hidden_dim, oup, kernel_size=1, relu=False),
        )

        if stride == 1 and inp == oup:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                depthwise_conv(inp, inp, kernel_size, stride, relu=False),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class VGG_ghost_v2(nn.Module):
    def __init__(self, vgg_name):
        super(VGG_ghost_v2, self).__init__()
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
                output_channel = _make_divisible(64 * width_mult, 4)
                hidden_channel = _make_divisible(64 * width_mult, 4)
                layers.append(block(in_channels, hidden_channel, output_channel, kernel_size=3, stride=1, use_se=0))
                in_channels = 64
            elif x == 'bottleneck_128':
                output_channel = _make_divisible(128 * width_mult, 4)
                hidden_channel = _make_divisible(128 * width_mult, 4)
                layers.append(block(in_channels, hidden_channel, output_channel, kernel_size=3, stride=1, use_se=0))
                in_channels = 128
            elif x == 'bottleneck_256':
                output_channel = _make_divisible(256 * width_mult, 4)
                hidden_channel = _make_divisible(256 * width_mult, 4)
                layers.append(block(in_channels, hidden_channel, output_channel, kernel_size=3, stride=1, use_se=0))
                in_channels = 256
            elif x == 'bottleneck_512':
                output_channel = _make_divisible(512 * width_mult, 4)
                hidden_channel = _make_divisible(512 * width_mult, 4)
                layers.append(block(in_channels, hidden_channel, output_channel, kernel_size=3, stride=1, use_se=0))
                in_channels = 512
            else:
                layers += [GhostModule(in_channels, _make_divisible(x * width_mult, 4), kernel_size=3, relu=True)]
                in_channels = _make_divisible(x * width_mult, 4)
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

