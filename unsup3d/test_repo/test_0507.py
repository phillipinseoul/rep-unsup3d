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
# print(np.array(depth_map.detach()))
# print(depth_map[0][0][63][63])

depth_map_n = np.array(depth_map.detach())[0][0]
normal_map = depth_map_n

for i, j in zip(range(1, 63), range(1, 63)):
    v1 = np.array([i, j - 1, depth_map_n[i][j - 1]], dtype='float64')
    v2 = np.array([i - 1, j, depth_map_n[i - 1][j]], dtype='float64')
    c = np.array([i, j, depth_map_n[j][i]], dtype='float64')
    d = np.cross(v2 - c, v1 - c)
    n = d / np.sqrt((np.sum(d ** 2)))
    normal_map[j][i] = n

cv2.imwrite('normal.png', normal_map * 255)

