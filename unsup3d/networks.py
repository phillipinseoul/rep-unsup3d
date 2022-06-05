'''
Define each network for the Photo-geometric Autoencoding pipeline.
'''
import torch.nn as nn
import torch
import torch.optim as optims

from unsup3d.modules import Encoder, AutoEncoder, Conf_Conv
from unsup3d.__init__ import *


# Image decompostion pipline
class ImageDecomp(nn.Module):
    def __init__(self, device, W, H,
                 depth_v, alb_v, light_v, view_v, use_conf, use_light):
        super(ImageDecomp,self).__init__()
        if depth_v == 'depth_v0':
            self.depth_net = AutoEncoder(cout=1, no_activate = True).to(device)        # B x 1 x W x H
        if alb_v == 'alb_v0':
            self.alb_net = AutoEncoder(cout=3).to(device)          # B x 3 x W x H
        if light_v == 'light_v0':
            self.light_net = Encoder(cout=4).to(device)            # B x 4
        if view_v == 'view_v0':
            self.view_net = Encoder(cout=6).to(device)             # B x 4 x 1 x 1

        ''' TODO: additional networks '''
        depth_pad = torch.zeros(1,H,W-4).to(device)
        self.depth_border = nn.functional.pad(depth_pad, (2,2), mode = 'constant', value = 1.0)
        self.border_depth = 0.7 * 1.1 + 0.3 * 0.9
        self.use_light = use_light
        self.use_conf = use_conf

        if self.use_conf:
            self.conf_net = Conf_Conv().to(device)
        else:
            self.conf_large = torch.ones(1,2,H,W, dtype=torch.float32).to(device)
            self.conf_small = torch.ones(1,2,H//4,W//4, dtype=torch.float32).to(device)

        if not use_light:
            self.shade_net = AutoEncoder(cout = 1).to(device)

    def get_depth_map(self, input):
        raw_res = self.depth_net(input)
        raw_res = raw_res - torch.mean(raw_res, dim = (1,2,3), keepdim=True)        # normalize
        res = raw_res.tanh()

        res = 1.0 + res / 10.0
        res = res * (1 - self.depth_border) + self.depth_border * self.border_depth  # border clamping

        return res
    
    def get_albedo(self, input):
        res = self.alb_net(input)
        res = (res + 1.)/ 2.
        return res

    def get_light(self, input):
        return self.light_net(input).squeeze(-1).squeeze(-1)

    def get_view(self, input):
        return self.view_net(input).squeeze(-1).squeeze(-1)

    def get_confidence(self, input):
        if self.use_conf:
            return self.conf_net(input)
        else:
            B = input.shape[0]
            return self.conf_small.repeat(B,1,1,1), self.conf_large.repeat(B,1,1,1)

    def get_shade(self, input):
        res = self.shade_net(input)
        return (res + 1.)/2.