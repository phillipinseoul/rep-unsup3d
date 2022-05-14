import torch
import os
import cv2
from torch.utils.data import DataLoader, Dataset
import os.path as path

MAX_PIX = 255
setting_list = ['train','val','test']
CelebA_PATH = '/root/unsup3d-rep/data/celeba'

class CelebA(Dataset):
    def __init__(self, setting = "train", img_size = (64,64)):
        '''check setting first'''
        if setting not in setting_list:
            print("CelebA, wrong data setting, you should select one of 'train', 'test' or 'val'.")
            print("what you selected : ", setting)
            assert(0)
        

        self.path = path.join(CelebA_PATH, setting)
        self.file_list = [name for name in os.listdir(self.path) if path.isfile(path.join(self.path,name))]
        self.img_size = img_size

    def __getitem__(self, idx):
        '''
        return image, changed as tensor.
        As original image is 128 x 128, we need to resize it as 64 by 64 here.

        the output will be scaled as '0~1' as torch.float32
        '''
        img = cv2.imread(path.join(self.path, self.file_list[idx]))
        re_img = cv2.resize(img, self.img_size, interpolation = cv2.INTER_LINEAR)
        re_img = torch.tensor(re_img, dtype = torch.float32)
        re_img = re_img.permute(2,0,1)                  # 3 x H x W
        re_img /= MAX_PIX                               # change value range 0~1

        return re_img


    def __len__(self):
        return len(self.file_list)