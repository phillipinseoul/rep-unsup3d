'''
Define each network for the Photo-geometric Autoencoding pipeline.
'''
import torch.nn as nn
import torch

from unsup3d.modules import Encoder, AutoEncoder, Conf_Conv



# Image decompostion pipline
class ImageDecomp(nn.Module):
    def __init__(self, device,
                 depth_v, alb_v, light_v, view_v, use_conf):
        super(ImageDecomp,self).__init__()
        if depth_v == 'depth_v0':
            self.depth_net = AutoEncoder(cout=1).to(device)        # B x 1 x W x H
        if alb_v == 'alb_v0':
            self.alb_net = AutoEncoder(cout=3).to(device)          # B x 3 x W x H
        if light_v == 'light_v0':
            self.light_net = Encoder(cout=4).to(device)            # B x 4
        if view_v == 'view_v0':
            self.view_net = Encoder(cout=6).to(device)             # B x 4 x 1 x 1

        ''' TODO: additional networks '''

        if use_conf:
            self.conf_net = Conf_Conv().to(device)

    def get_depth_map(self, input):
        res = self.depth_net(input)
        res = 1.0 + res/10.0
        return res
    
    def get_albedo(self, input):
        res = self.alb_net(input)
        res = (res + 1.0)/ 2
        return res

    def get_light(self, input):
        return self.light_net(input).squeeze(-1).squeeze(-1)

    def get_view(self, input):
        return self.view_net(input).squeeze(-1).squeeze(-1)

    def get_confidence(self, input):
        return self.conf_net(input)