from unsup3d.train import Trainer
from unsup3d.dataloader import CelebA
from unsup3d.renderer import *
from unsup3d.utils import ImageFormation
from unsup3d.model import PhotoGeoAE, PercepLoss
from unsup3d.renderer import RenderPipeline
from unsup3d.modules import AutoEncoder, Encoder
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optims
import numpy as np
from PIL import Image
import PIL
import yaml
from tensorboardX import SummaryWriter

def img_to_input(img_path, image_size=(64, 64)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    input = Image.open(img_path).convert('RGB')
    input = transform(input).unsqueeze(0)
    print(f"input shape: {input.shape}")
    
    return input

class SimpleAE(nn.Module):
    def __init__(self, device):
        super(SimpleAE, self).__init__()
        self.lambda_p = 0.5
        self.lambda_f = 1.0
        self.get_depth_map = AutoEncoder(1)
        self.get_albedo = AutoEncoder(3)
        self.get_light = Encoder(4)
        self.get_view = Encoder(6)
        self.use_gt_depth = False
        self.percep = PercepLoss()
        self.imgForm = ImageFormation(size=64)
        self.render = RenderPipeline(
            b_size = 1,
            device=device
        )
        self.learning_rate = 1e-4

        '''concat all parameters'''
        all_params = list(self.get_depth_map.parameters()) + list(self.get_albedo.parameters()) \
                        + list(self.get_light.parameters()) + list(self.get_view.parameters())

        self.optimizer = optims.Adam(
            params = all_params,
            lr = self.learning_rate
        )

    def forward(self, input):
        depth = self.get_depth_map(input)
        albedo = self.get_albedo(input)
        view = self.get_view(input).squeeze(-1).squeeze(-1)
        light = self.get_light(input)

        normal = self.imgForm.depth_to_normal(depth)
        shading = self.imgForm.normal_to_shading(normal, light)
        canon_img = self.imgForm.alb_to_canon(albedo, shading)

        org_img, org_depth = self.render(
            canon_depth=depth.cuda(), 
            canon_img=canon_img.cuda(), 
            views=view.cuda()
        )

        return org_img, org_depth

def get_photo_loss(img1, img2):
    L1_loss = torch.abs(img1 - img2)
    return torch.sum(L1_loss)
    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter('test_results/logs')

    # load a single input image
    celeba_path = 'data/celeba_ex/001_face'
    input = img_to_input(os.path.join(celeba_path, 'input_image.png'))  # B x 3 x W x H

    input_x = input.squeeze(0)
    writer.add_image('input/input_image', input_x)
    input = input.to(device)

    simpleAE = SimpleAE(device)

    num_epochs = 200
    for i in range(num_epochs):
        input = input.detach().cpu()
        recon_img, recon_depth = simpleAE(input)

        input = input.to(device)
        loss = get_photo_loss(recon_img, input)
        simpleAE.optimizer.zero_grad()
        loss.backward()
        simpleAE.optimizer.step()

        recon_img_x = recon_img.squeeze(0)
        writer.add_image('recon/recon_image', recon_img_x, i)
        print(f"iter {i}: loss={loss.item()}")
        writer.add_scalar('loss/photo_loss', loss.item(), i)


'''
$ tensorboard --logdir test_results/logs --port 6001
'''


