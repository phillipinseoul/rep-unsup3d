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
from modules import AutoEncoder

IMAGE_SIZE = 64

albnet = AutoEncoder(3)

img = Image.open('./conan.png').convert('RGB')

transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            ])

input = transform(img)
print(input.shape)
input = input.unsqueeze(0)
print(input.shape)

# loss function & optimizer
loss_fn = nn.L1Loss()
optimizer = optim.SGD(albnet.parameters(), lr=0.001, momentum=0.9)

for i in range(1000):
    loss = 0
    optimizer.zero_grad()
    output = albnet(input)
    print(output.shape)

    loss = loss_fn(output, input)

    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(loss.item())
        output = output.squeeze()
        toPILImage = ToPILImage() 
        albedo_img = toPILImage(output)
        plt.imshow(albedo_img)
        # plt.show()
        plt.savefig('albedo.png')

'''
predicted_albedo = albnet(input)
predicted_albedo = predicted_albedo.squeeze()

toPILImage = ToPILImage() 
albedo_img = toPILImage(predicted_albedo)

plt.imshow(albedo_img)
# plt.show()
plt.savefig('albedo_img.png')
'''