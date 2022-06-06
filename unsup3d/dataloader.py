from mimetypes import init
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import cv2
from torch.utils.data import DataLoader, Dataset
import os.path as path
from unsup3d.__init__ import *

MAX_PIX = 255
setting_list = ['train','val','test']
CelebA_PATH = '/root/unsup3d-rep/data/celeba'
BFM_PATH = '/root/unsup3d-rep/data/synface'


class CelebA(Dataset):
    def __init__(self, setting = "train", img_size = 64, w_perturb = False):
        '''check setting first'''
        if setting not in setting_list:
            print("CelebA, wrong data setting, you should select one of 'train', 'test' or 'val'.")
            print("you have selected : ", setting)
            assert(0)
        
        self.path = path.join(CelebA_PATH, setting)
        self.is_train = True if setting == "train" else False
        self.file_list = [name for name in os.listdir(self.path) if path.isfile(path.join(self.path,name))]
        self.img_size = (img_size, img_size)

        self.WITH_PERTURB = w_perturb

    def __getitem__(self, idx):
        '''
        return image, changed as tensor.
        As original image is 128 x 128, we need to resize it as 64 by 64 here.

        the output will be scaled as '0~1' as torch.float32
        '''
        img = cv2.imread(path.join(self.path, self.file_list[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        re_img = cv2.resize(img, self.img_size, interpolation = cv2.INTER_LINEAR)
        re_img = torch.tensor(re_img, dtype = torch.float32)
        re_img = re_img.permute(2,0,1)                  # 3 x H x W
        re_img /= MAX_PIX                               # change value range 0~1
        
        if self.is_train and np.random.rand() > 0.5:
            re_img = torch.flip(re_img, dims = [2])

        if self.WITH_PERTURB:
            re_img = asym_perturb(re_img)

        return re_img

    def __len__(self):
        #return 32*2
        return len(self.file_list)


class BFM(Dataset):
    def __init__(self, setting = "train", img_size = 64, crop_rate = 0.125, w_perturb = False):
        '''check setting first'''
        if setting not in setting_list:
            print("BFM, wrong data setting, you should select one of 'train', 'test' or 'val'.")
            print("you have selected : ", setting)
            assert(0)
        
        self.is_train = True if setting == "train" else False

        self.WITH_PERTURB = w_perturb
        self.crop = 170
        # self.crop = 192
        
        self.path = path.join(BFM_PATH, setting)
        self.img_path = path.join(self.path, 'image')       # path for images
        self.gt_path = path.join(self.path, 'depth')        # path for ground truth depth maps
        self.img_size = (img_size, img_size)
        self.crop_rate = crop_rate

        img_list = [name for name in os.listdir(self.img_path) if path.isfile(path.join(self.img_path, name))]
        gt_list = [name for name in os.listdir(self.gt_path) if path.isfile(path.join(self.gt_path, name))]
        img_list.sort()
        gt_list.sort()

        '''make (image, gt_depth) pairs'''
        img_gt_pairs = []
        for image, gt_depth in zip(img_list, gt_list):
            assert image.replace('image', 'depth') == gt_depth, 'image and gt_depth do not match'
            img_gt_pairs.append((image, gt_depth))

        self.img_gt_pairs = img_gt_pairs

    def transform(self, img, hflip=False):
        if self.crop is not None:
            if isinstance(self.crop, int):
                img = transforms.CenterCrop(self.crop)(img)
            else:
                assert len(self.crop) == 4, 'Crop size must be an integer for center crop, or a list of 4 integers (y0,x0,h,w)'
                img = transforms.functional.crop(img, *self.crop)
        img = transforms.functional.resize(img, (64, 64))
        if hflip:
            img = transforms.functional.hflip(img)
        return transforms.functional.to_tensor(img)

    def __getitem__(self, index):
        path_A = path.join(self.img_path, self.img_gt_pairs[index][0])
        path_B = path.join(self.gt_path, self.img_gt_pairs[index][1])
        img_A = Image.open(path_A).convert('RGB')
        img_B = Image.open(path_B).convert('RGB')
        hflip = self.is_train and (np.random.rand() > 0.5)

        img = self.transform(img_A, hflip=hflip)

        if self.WITH_PERTURB:
           img = asym_perturb(img)
        
        depth = self.transform(img_B, hflip=hflip)
        depth = (1-depth) * 0.2 + 0.9
        
        return img, depth[0:1,:,:]

    def __len__(self):
        return len(self.img_gt_pairs)
        # return 64 * 100


def asym_perturb(image_tensor):
    '''
    input:
    - image_tensor : 3 x H x W (0~1)
    return
    - res : 3 x H x W (0~1)
    '''
    # We will call it during 
    _, H, W = image_tensor.shape
    random_color = torch.rand(3,1,1)
    rand_alpha = np.random.rand(1)*(1.0 - 0.5) + 0.5    # transparency
    rand_patch_size = np.random.rand(2)*(0.5-0.2) + 0.2

    patch_H = int(rand_patch_size[0] * H)
    patch_W = int(rand_patch_size[1] * W)

    random_patch = get_rand_patch(H, W, patch_H, patch_W)
    
    alpha_patch = random_patch * rand_alpha
    bg = image_tensor * (1. - alpha_patch)
    fg = random_patch * random_color * alpha_patch

    res = bg + fg

    return res.float()



def get_rand_patch(H, W, P_H, P_W):
    bg = torch.zeros(3, H, W)

    rand_y = np.random.randint(0, H - P_H + 1)
    rand_x = np.random.randint(0, W - P_W + 1)

    bg[:,rand_y:(rand_y+P_H), rand_x:(rand_x+P_W)] = 1.0

    return bg




'''
    def __getitem__(self, idx):
        #return both image and gt depth_map as tensor.

        #crop settings
        top = int(self.img_size[0] * self.crop_rate)            # crop out `crop_rate` of top, bottom, left, right
        bottom = int(self.img_size[0] * (1-self.crop_rate))

        #resize image
        img = cv2.imread(path.join(self.img_path, self.img_gt_pairs[idx][0]))

        re_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        re_img = cv2.resize(re_img, self.img_size, interpolation = cv2.INTER_LINEAR)
        re_img = re_img[top:bottom, top:bottom]                     # cropping
        re_img = cv2.resize(re_img, self.img_size, interpolation = cv2.INTER_LINEAR)
        re_img = torch.tensor(re_img, dtype = torch.float32)
        re_img = re_img.permute(2, 0, 1)                    # 3 x H x W
        re_img /= MAX_PIX                                   # change value range 0~1

        #resize gt depth map
        gt_depth = cv2.imread(path.join(self.gt_path, self.img_gt_pairs[idx][1]))
        
        #re_depth = cv2.cvtColor(gt_depth, cv2.COLOR_BGR2GRAY)
        re_depth = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        re_depth = cv2.resize(re_depth, self.img_size, interpolation = cv2.INTER_LINEAR)
        re_depth = re_depth[top:bottom, top:bottom]                 # add image cropping
        re_depth = cv2.resize(re_depth, self.img_size, interpolation = cv2.INTER_LINEAR)
        re_depth = torch.tensor(re_depth, dtype = torch.float32)#.unsqueeze(-1)
        re_depth = re_depth.permute(2, 0, 1)                  # 1 x H x W
        re_depth /= MAX_PIX                                   # change value range 0~1
        re_depth = (1 - re_depth) * 0.2 + 0.9                 # depth: 0.9 ~1.1
        re_depth = re_depth[0:1,:,:]


        if np.random.rand() > 0.5:
            re_img = transforms.functional.hflip(re_img)
            re_depth = transforms.functional.hflip(re_depth)

        if self.WITH_PERTURB:
           re_img = asym_perturb(re_img)
        
        return re_img, re_depth
    '''