from unsup3d.train import Trainer
#from unsup3d.model import PercepLoss
from unsup3d.dataloader import CelebA
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN,self).__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),            
        )

    def forward(self, inp):
        res = self.enc(inp)
        res = nn.functional.softmax(res)

        loss = torch.mean(torch.abs(res - inp), dim = (0,1,2,3))    # test with L1 loss

        return loss



def test_image(path):
    img = mpimg.imread(path)
    imgplot = plt.imshow(img)
    
    print("img size :",img.shape)

    return img

def get_torch_image(b_size, H, W, path):
    img = test_image(path)
    img = torch.tensor(img/255, dtype = torch.float32)
    img = img.permute(2,0,1)    # set channel first
    transform = nn.Sequential(
        transforms.Resize((H,W)) ,
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    )
    out = transform(img)
    out = out.unsqueeze(0)
    out = out.repeat(b_size,1,1,1)
    
    return out


def test_percep_0508():
    per = PercepLoss()
    path1 = "./test1.jpg"   
    path2 = "./test2.jpg"
    imgs1 = get_torch_image(13, 64, 64, path1)
    imgs2 = get_torch_image(13, 64, 64, path2)

    rand_conf = torch.abs(torch.rand(13,1,16,16))


    loss = per(imgs1, imgs2, rand_conf)

    print(loss)
    

def test_celeba_dataloader_0509():
    setting_list = ['train','val','test']
    for set in setting_list:
        CA = CelebA(setting = set)
        length =  CA.__len__()
        print("len ", set, " type dataset: ", length)

        for j in range(50):
            ind = torch.randint(0,length-1,(1,1))
            a = CA.__getitem__(ind.item())
            print(a.shape)

def test_Trainer_0509():
    simplenn = SimpleNN()
    trainer = Trainer(None, model = simplenn)
    trainer.train()


if __name__ == '__main__':
    test_Trainer_0509()



