import torch
import torch.nn as nn
from torch.autograd import Function
from .resnet import _resnet
from .resnet import Bottleneck as Bottleneck_default
from torch.autograd import Variable
# from .pose_net import *


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class DomainClassifier(nn.Module):
    def __init__(self, input_dim=256, ndf=64, with_bias=False):
        super(DomainClassifier, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, ndf, kernel_size=4, stride=2, padding=1, bias=with_bias)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=with_bias)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=with_bias)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=with_bias)
        self.conv5 = nn.Conv2d(ndf * 8, ndf * 16, kernel_size=4, stride=2, padding=1, bias=with_bias)
        self.conv6 = nn.Conv2d(ndf * 16, 1, kernel_size=2, stride=1, bias=with_bias)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.conv5(x)
        x = self.leaky_relu(x)
        x = self.conv6(x)
        return x

def get_default_network_config():
    config = edict()
    config.from_model_zoo = True
    config.pretrained = ''
    config.num_layers = 101
    config.num_deconv_layers = 4
    config.num_deconv_filters = 256
    config.num_deconv_kernel = 4
    config.final_conv_kernel = 1
    config.depth_dim = 1
    config.input_channel = 3
    return config

class DeconvHead(nn.Module):
    def __init__(self, in_channels, num_layers, num_filters, kernel_size, conv_kernel_size, num_joints, depth_dim,
                 with_bias_end=True):
        super(DeconvHead, self).__init__()

        conv_num_filters = num_joints * depth_dim

        assert kernel_size == 2 or kernel_size == 3 or kernel_size == 4, 'Only support kenerl 2, 3 and 4'
        padding = 1
        output_padding = 0
        if kernel_size == 3:
            output_padding = 1
        elif kernel_size == 2:
            padding = 0

        assert conv_kernel_size == 1 or conv_kernel_size == 3, 'Only support kenerl 1 and 3'
        if conv_kernel_size == 1:
            pad = 0
        elif conv_kernel_size == 3:
            pad = 1

        self.features = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                _in_channels = in_channels
                self.features.append(
                    nn.Conv2d(_in_channels, num_filters, kernel_size=1, stride=1, bias=False))
                self.features.append(nn.BatchNorm2d(num_filters))
                self.features.append(nn.ReLU(inplace=True))
            else:
                _in_channels = num_filters
                self.features.append(
                    nn.ConvTranspose2d(_in_channels, num_filters, kernel_size=kernel_size, stride=2, padding=padding,
                                       output_padding=output_padding, bias=False))
                self.features.append(nn.BatchNorm2d(num_filters))
                self.features.append(nn.ReLU(inplace=True))

        if with_bias_end:
            self.features.append(
                nn.Conv2d(num_filters, conv_num_filters, kernel_size=conv_kernel_size, padding=pad, bias=True))
        else:
            self.features.append(
                nn.Conv2d(num_filters, conv_num_filters, kernel_size=conv_kernel_size, padding=pad, bias=False))
            self.features.append(nn.BatchNorm2d(conv_num_filters))
            self.features.append(nn.ReLU(inplace=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)

    def forward(self, x):
        features = []
        for i, l in enumerate(self.features):
            x = l(x)
            if (i+1) % 3 == 0:
                features.append(x)
        return x, features

