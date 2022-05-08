import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from modules import AutoEncoder, Encoder

IMAGE_SIZE = 64

img = Image.open('./conan.png').convert('RGB')

transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            ])

input = transform(img)
input = input.unsqueeze(0)
print(f'input shape: {input.shape}')

# define decomposition networks
alb_net = AutoEncoder(3)
depth_net = AutoEncoder(1)
view_net = Encoder(6)
light_net = Encoder(4)

# derive normal map
depth_map = depth_net(input)
print(f'depth_map: {depth_map.shape}')
print(depth_map)

