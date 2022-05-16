'''
Define each network for the Photo-geometric Autoencoding pipeline.
'''
from unsup3d.modules import Encoder, AutoEncoder, Conf_Conv

# Image decompostion pipline
class ImageDecomp():
    def __init__(self, depth_v='v0', alb_v='v0', light_v='v0', view_v='v0'):
        if depth_v == 'v0':
            self.depth_net = AutoEncoder(1)         # B x 1 x W x H
        
        ''' TODO: additional depth networks'''
        
        if alb_v == 'v0':
            self.alb_net = AutoEncoder(3)           # B x 3 x W x H

        ''' TODO: additional albedo networks'''

        if light_v == 'v0':
            self.light_net = Encoder(4)             # B x 4 x 1 x 1
        
        ''' TODO: additional light networks '''
        
        if view_v == 'v0':
            self.view_net = Encoder(6)              # B x 4 x 1 x 1

        ''' TODO: additional view networks '''

    def get_depth_map(self, input):
        return self.depth_net(input)
    
    def get_albedo(self, input):
        return self.alb_net(input)

    def get_light(self, input):
        return self.light_net(input)

    def get_view(self, input):
        return self.view_net(input)


class ConfNet_v1():
    def __init__(self):
        pass


class Renderer_v1():       # is it nn.module ??? (04/25)
    def __init__(self):
        pass