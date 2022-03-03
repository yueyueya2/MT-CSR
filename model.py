import torch
import time
import os
import numpy as np
import pandas as pd
from torch import nn, optim

from cell import *
from parameters import *

##################### completion network #####################
class CMPNet(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, timeslot, feature, poi):
        super(CMPNet, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.timeslot = timeslot
        self.feature = feature
        self.poi = poi
        layers = []
        for num in range(self.timeslot):
            layers.append(AdaCNN(self.in_channel, self.out_channel, self.kernel_size, self.feature))
        self.layers = nn.ModuleList(layers)
        self.fea_comp = AuxCMP(self.poi)
        self.weight = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, x, m):
        # raw covcmp
        bsize, time, channel, shape1, shape2 = x.shape
        for t in range(time):
           if t == 0:
               h = self._ini_hidden(bsize, channel, shape1, shape2)
           input_x = x[:, t, :, :, :]
           output1 = self.layers[t](h, input_x)
           h = output1
        #raw auxcmp
        output2 = self.fea_comp(x, m)

        # cmpsr-cos
        output = self.weight * output1 + (1 - self.weight) * output2
        return output

    def _ini_hidden(self, bs, c, w, h):
        # print(bs, c, w, h)
        return torch.zeros((bs, c, w, h)).to(device)


##################### super resolution network #####################
class SRNet_cell(nn.Module):
    def __init__(self, in_channel, channel_n, scaler_n, resnet_n, step, weather=whether_weather):
        super(SRNet_cell, self).__init__()
        self.step = step
        self.channel_n = channel_n
        self.weight = torch.tensor([0.1, 0.3, 0.6], dtype=torch.float32)
        # ResExt
        self.exnet = Resnet(channel_n, resnet_n)
        self.upnet = UpscaleBlock(channel_n, scaler_n)
        # revise channel 2 to 32
        if weather == False:
            self.ch1 = nn.Sequential(
                # add feature channel: in_channel + 1
                nn.Conv2d(in_channel, channel_n, 3, 1, 1),
                nn.ReLU(inplace=True),
            )
        else:
            self.ch1 = nn.Sequential(
                # add feature channel: in_channel + 1
                nn.Conv2d(in_channel + 1, channel_n, 3, 1, 1),
                nn.ReLU(inplace=True),
            )
        # revise channel 32 to 2
        self.ch2 = nn.Sequential(
            nn.Conv2d(channel_n, in_channel, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, feature_reshape=None):
        if feature_reshape is None:
            outputs = 0
            for s in range(self.step):
                if s == 0:
                    x0 = x
                    x0 = self.ch1(x0)
                x1 = self.exnet(x0)
                x2 = self.upnet(x1)
                outputs += self.weight[s] * self.ch2(x2)
                x0 = x1
        else:
            outputs = 0
            for s in range(self.step):
                if s == 0:
                    x0 = torch.cat((x, feature_reshape), dim=1)
                    x0 = self.ch1(x0)
                x1 = self.exnet(x0)
                x2 = self.upnet(x1)
                outputs += self.weight[s] * self.ch2(x2)
                x0 = x1
        return outputs


class SRNet(nn.Module):
    def __init__(self, in_channel, channel_n, scaler_n, resnet_n, step):
        super(SRNet, self).__init__()
        # upscale times
        self.block_n = int(scaler_n / 2)
        self.extf = FeaExtraction()
        self.SRcell = SRNet_cell(in_channel, channel_n, scaler_n, resnet_n, step)
        self.upsacle = UpscaleBlock(channel_n, 2)

        # upscale feature
        self.srf = nn.Sequential(
            nn.Conv2d(1, 4, 3, 1, 1),
            nn.BatchNorm2d(4),
            nn.PixelShuffle(upscale_factor=2),
            nn.ReLU(inplace=True)
        )

        # distribution upsampling
        self.distribution = Distribution_upsampling(scaler_n)
        # combine lr with hr
        self.combine = combine_with_distribution(scaler_n)

    def forward(self, x, feature=None):
        # x - (n, 2, 16, 16); feature - (n, type)
        if feature is None:
            for t in range(self.block_n):
                # initialize the first upsampling block
                if t == 0:
                    x0 = x
                x1 = self.SRcell(x0)
                x0 = x1
            out = x1
            out = self.distribution(out)
            out = self.combine(out, x)
        else:
            for t in range(self.block_n):
                # initialize the first upsampling block
                if t == 0:
                    x0 = x
                    e0 = self.extf(feature)
                x1 = self.SRcell(x0, e0)
                # output as next block's input
                # upsample external feature at the same time
                x0 = x1
                e0 = self.srf(e0)
            out = x1
            out = self.distribution(out)
            out = self.combine(out, x)
        return out


