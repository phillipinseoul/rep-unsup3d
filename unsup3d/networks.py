'''

'''
from .modules import Encoder, AutoEncoder, Conf_Conv


class DepthNet_v1():
    def __init__(self):
        self.depth_net = AutoEncoder(1)       # AutoEncoder, cout=1

    def depth_to_normal(self):
        pass

    def normal_to_shading(self):
        pass

'''
class DepthNet_v2():
    def __init__(self):
        pass
'''

'''
    Should we define the renderer functions inside these classes,
    instead of inside the Renderer() class?
'''


class AlbedoNet_v1():
    def __init__(self):
        self.alb_net = AutoEncoder(3)       # AutoEncoder, cout=3

    def alb_to_canon(self):
        pass


class LightNet_v1():
    def __init__(self):
        self.light_net = Encoder(4)       # Encoder, cout=4


class ViewNet_v1():
    def __init__(self):
        self.view_net = Encoder(6)       # Encoder, cout=6


class ConfNet_v1():
    def __init__(self):
        pass


class Renderer_v1():       # is it nn.module ??? (04/25)
    def __init__(self):
        pass