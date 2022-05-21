import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
import cv2
from torch.utils.data import DataLoader, Dataset
import os.path as path

MAX_PIX = 255
setting_list = ['train','val','test']
CelebA_PATH = '/root/unsup3d-rep/data/celeba'
BFM_PATH = '/root/unsup3d-rep/data/synface'

class CelebA(Dataset):
    def __init__(self, setting = "train", img_size = 64):
        '''check setting first'''
        if setting not in setting_list:
            print("CelebA, wrong data setting, you should select one of 'train', 'test' or 'val'.")
            print("you have selected : ", setting)
            assert(0)
        
        self.path = path.join(CelebA_PATH, setting)
        self.file_list = [name for name in os.listdir(self.path) if path.isfile(path.join(self.path,name))]
        self.img_size = (img_size, img_size)

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


class BFM(Dataset):
    def __init__(self, setting = "train", img_size = 64):
        '''check setting first'''
        if setting not in setting_list:
            print("BFM, wrong data setting, you should select one of 'train', 'test' or 'val'.")
            print("you have selected : ", setting)
            assert(0)

        self.path = path.join(BFM_PATH, setting)
        self.img_path = path.join(self.path, 'image')       # path for images
        self.gt_path = path.join(self.path, 'depth')        # path for ground truth depth maps
        self.img_size = (img_size, img_size)

        img_list = [name for name in os.listdir(self.img_path) if path.isfile(path.join(self.img_path, name))].sort()
        gt_list = [name for name in os.listdir(self.gt_path) if path.isfile(path.join(self.gt_path, name))].sort()

        '''make (image, gt_depth) pairs'''
        img_gt_pairs = []
        for image, gt_depth in zip(img_list, gt_list):
            assert img.replace('image', 'depth') == gt_depth, 'image and gt_depth do not match'
            img_gt_pairs.append((image, gt_depth))

        self.img_gt_pairs = img_gt_pairs

    def __getitem__(self, idx):
        '''return both image and gt depth_map as tensor.'''

        '''resize image'''
        img = cv2.imread(path.join(self.img_path, self.img_gt_pairs[idx][0]))
        re_img = cv2.resize(img, self.img_size, interpolation = cv2.INTER_LINEAR)
        re_img = torch.tensor(re_img, dtype = torch.float32)
        re_img = re_img.permute(2, 0, 1)                    # 3 x H x W
        re_img /= MAX_PIX                                   # change value range 0~1

        '''resize gt depth map'''
        gt_depth = cv2.imread(path.join(self.gt_path, self.img_gt_pairs[idx][1]))
        re_depth = cv2.resize(gt_depth, self.img_size, interpolation = cv2.INTER_LINEAR)
        re_depth = cv2.cvtColor(re_depth, cv2.COLOR_BGR2GRAY)
        re_depth = torch.tensor(gt_depth, dtype = torch.float32).unsqueeze(-1)
        re_depth = re_img.permute(2, 0, 1)                  # 1 x H x W
        re_depth /= MAX_PIX                                 # change value range 0~1
        
        return re_img, re_depth
    
    def __len__(self):
        return len(self.img_gt_pairs)