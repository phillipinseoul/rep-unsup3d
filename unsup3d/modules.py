'''
Define all neural networks used in photo-geometric pipeline
Any learnable parameters shouldn't be defined out of this file
'''
import os
import requests
import torch
import torch.nn as nn
from unsup3d.__init__ import *


# network architecture for viewpoint, lighting
class Encoder(nn.Module):
    def __init__(self, cout):
        '''
        * view: cout = 6
        * lighting: cout = 4
        '''
        super(Encoder, self).__init__()

        # encoder network
        encoder = [
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, cout, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        ]
        self.encoder = nn.Sequential(*encoder)

    def forward(self, input):
        out = self.encoder(input)

        return out

# network architecture for depth, albedo
class AutoEncoder(nn.Module):
    def __init__(self, cout, no_activate = False):
        '''
        * depth: cout=1
        * albedo: cout=3
        '''
        super(AutoEncoder, self).__init__()

        # test differentiable activation function instead of ReLU (06/01)
        if test_ELU:
            # encoder network
            encoder = [
                nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),       # layer 1
                nn.GroupNorm(16, 64),
                nn.ELU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),      # layer 2
                nn.GroupNorm(32, 128),
                nn.ELU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),      # layer 3
                nn.GroupNorm(64, 256),
                nn.ELU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),      # layer 4
                nn.ELU(inplace=True),
                nn.Conv2d(512, 256, kernel_size=4, stride=1, padding=0, bias=False),      # layer 5
                nn.ELU(inplace=True),
            ]

            # decoder network
            decoder = [
                nn.ConvTranspose2d(256, 512, kernel_size=4, stride=1, padding=0, bias=False),
                nn.ELU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.GroupNorm(64, 256),
                nn.ELU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(64, 256),
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
                nn.GroupNorm(32, 128),
                nn.ELU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(32, 128),
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
                nn.GroupNorm(16, 64),
                nn.ELU(inplace=True),

                nn.Upsample(scale_factor=2, mode = 'nearest'),
                
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(16, 64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2, bias=False),
                nn.GroupNorm(16, 64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, cout, kernel_size=5, stride=1, padding=2, bias=False)
            ]
        else:
            # encoder network
            encoder = [
                nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),       # layer 1
                nn.GroupNorm(16, 64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),      # layer 2
                nn.GroupNorm(32, 128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),      # layer 3
                nn.GroupNorm(64, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),      # layer 4
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(512, 256, kernel_size=4, stride=1, padding=0, bias=False),      # layer 5
                nn.ReLU(inplace=True)
            ]

            # decoder network
            decoder = [
                nn.ConvTranspose2d(256, 512, kernel_size=4, stride=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.GroupNorm(64, 256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(64, 256),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
                nn.GroupNorm(32, 128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(32, 128),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
                nn.GroupNorm(16, 64),
                nn.ReLU(inplace=True),

                nn.Upsample(scale_factor=2, mode = 'nearest'),
                
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(16, 64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2, bias=False),
                nn.GroupNorm(16, 64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, cout, kernel_size=5, stride=1, padding=2, bias=False)
            ]

        if not no_activate:
            decoder.append(nn.Tanh())

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, input):
        out = self.encoder(input)
        out = self.decoder(out)

        return out

    
# network architecture for confidence map
class Conf_Conv(nn.Module):
    def __init__(self):
        '''
        `Conf_Conv` outputs two pairs of confidence maps at different
        spatial resolutions for i) photometric and ii) perceptual losses
        '''
        super(Conf_Conv, self).__init__()
        encoder = [
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, 64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(64, 256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 128, kernel_size=4, stride=1, padding=0, bias=False),
            nn.ReLU()
        ]

        decoder_1 = [
            nn.ConvTranspose2d(128, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(64, 256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.ReLU()
        ]

        out_1 = [
            nn.Conv2d(128, 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Softplus()
        ]

        decoder_2 = [
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, 64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, 64),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Softplus()
        ]

        self.encoder = nn.Sequential(*encoder)
        self.decoder_1 = nn.Sequential(*decoder_1)
        self.out_1 = nn.Sequential(*out_1)
        self.decoder_2 = nn.Sequential(*decoder_2)

    def forward(self, input):
        out = self.encoder(input)
        out = self.decoder_1(out)
        out_1 = self.out_1(out)
        out_2 = self.decoder_2(out)

        if torch_old:
            if out_1.isnan().sum() != 0 or out_2.isnan().sum() != 0:
                assert(0)
        
        return out_1, out_2




###########################################
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# https://github.com/facebookresearch/DeeperCluster/blob/main/src/model/vgg16.py
# This code is from FAIR's DeepCluster repository, with small modification
# We use this code to load pretrained file, for ablation test
# 

cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

class VGG16(nn.Module):
    '''
    VGG16 model 
    '''
    def __init__(self, dim_in, relu=True, dropout=0.5, batch_norm=True):
        super(VGG16, self).__init__()
        self.features = make_layers(cfg['D'], dim_in, batch_norm=batch_norm)
        self.dim_output_space = 4096
        classifier = [
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
        ]
        if relu:
            classifier.append(nn.ReLU(True))
        self.classifier = nn.Sequential(*classifier)

        load_pretrained_rotnet(self)
            
    def forward(self, x):
        x = self.features(x)
        if self.classifier is not None:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        return x


def make_layers(cfg, in_channels, batch_norm=True):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def load_pretrained_rotnet(model):
    save_name = './models/rotnet_imagenet.pth'
    url = 'https://dl.fbaipublicfiles.com/deepcluster/rotnet/rotnet_imagenet.pth'
    
    if not os.path.exists(save_name):
        r = requests.get(url, allow_redirects=True)
        open(save_name, 'wb').write(r.content)

    checkpoint = torch.load(save_name)
    checkpoint['state_dict'] = {rename_key(key): val
                                for key, val
                                in checkpoint['state_dict'].items()}

    if 'pred_layer.weight' in checkpoint['state_dict']:
        del checkpoint['state_dict']['pred_layer.weight']
        del checkpoint['state_dict']['pred_layer.bias']
    
    model.load_state_dict(checkpoint['state_dict'])
    print("succefully loaded")

def rename_key(key):
    "Remove module from key"
    if not 'module' in key:
        return key
    if key.startswith('module.body.'):
        return key[12:]
    if key.startswith('module.'):
        return key[7:]
    return ''.join(key.split('.module'))

    