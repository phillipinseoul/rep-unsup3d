import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

IMAGE_SIZE = 512

transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            ])

# changes a tensor to an image
toPILImage = ToPILImage()

# load input depth map (grayscale image)
depth = Image.open('test_imgs/face.png').convert("L")
depth = transform(depth)

# first, save the original depth map: `depth_map.png`
depth_map = toPILImage(depth)
plt.title('depth_map')
plt.imshow(depth_map)
plt.savefig('depth_map.png')

d, h, w = depth.shape
# print(depth.shape)

print(d)
print(h)
print(w)

# now, calculate normal map from depth map
normal_map = depth.repeat(3, 1, 1)

for i in range(1, IMAGE_SIZE - 1):
    for j in range(1, IMAGE_SIZE - 1):
        v1 = torch.Tensor([i, j - 1, depth[0][i][j - 1]])
        v2 = torch.Tensor([i - 1, j, depth[0][i - 1][j]])
        c = torch.Tensor([i, j, depth[0][i][j]])

        d = torch.cross(v2 - c, v1 - c)
        n = d / torch.sqrt(torch.sum(d ** 2))

        normal_map[:, i, j] = n

# save the normal map: `normal_map.png`
normal_map = toPILImage(normal_map)
plt.title('normal_map')
plt.imshow(normal_map)
plt.savefig('normal_map.png')