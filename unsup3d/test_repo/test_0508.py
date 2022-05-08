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
import cv2

IMAGE_SIZE = 64

# d_im = cv2.imread("depth.png")
# d_im = d_im.astype("float64")
# d_im = torch.Tensor(d_im)
d_im = Image.open('./depth.png').convert('RGB')
# ex = cv2.imread("./depth.png")

transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            ])

d_im = transform(d_im)
depth_map = d_im[0]
# normal_map = d_im.repeat(3, 1, 1)
normal_map = d_im

# print(normal_map.shape)


for i, j in zip(range(1, 63), range(1, 63)):
    v1 = np.array([i, j - 1, depth_map[i][j - 1]], dtype='float64')
    v2 = np.array([i - 1, j, depth_map[i - 1][j]], dtype='float64')
    c = np.array([i, j, depth_map[j][i]], dtype='float64')

    d = np.cross(v2 - c, v1 - c)
    n = d / np.sqrt((np.sum(d ** 2)))
    normal_map[:, j, i] = torch.Tensor(n)

normal_map *= 255
# print(normal_map)

n_map = np.array(d_im)
print(n_map.shape)

for i, j in zip(range(64), range(64)):
    n_map[:, j, i] = normal_map[:, j, i]

print(n_map)

'''
normal_map *= 255
print(normal_map)
toPILImage = ToPILImage() 
normal_map = toPILImage(normal_map)
plt.savefig('normal.png')

'''
cv2.imwrite('normal_map.png', n_map)
