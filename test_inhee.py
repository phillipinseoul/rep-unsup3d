from re import A
from unsup3d.train import Trainer
#from unsup3d.model import PercepLoss
from unsup3d.dataloader import CelebA
from unsup3d.renderer import *
from unsup3d.utils import ImageFormation
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
from PIL import Image 
import PIL 
from copy import deepcopy

'''
------------------------------- Author's renderer -------------------------------
'''

import torch
import math
#import neural_renderer as nr


cfgs = {}
EPS = 1e-7


class ARenderer():
    def __init__(self, cfgs):
        self.device = torch.device('cuda')
        self.image_size = cfgs.get('image_size', 64)
        self.min_depth = cfgs.get('min_depth', 0.9)
        self.max_depth = cfgs.get('max_depth', 1.1)
        self.rot_center_depth = cfgs.get('rot_center_depth', (self.min_depth+self.max_depth)/2)
        self.fov = cfgs.get('fov', 10)
        self.tex_cube_size = cfgs.get('tex_cube_size', 2)
        self.renderer_min_depth = cfgs.get('renderer_min_depth', 0.1)
        self.renderer_max_depth = cfgs.get('renderer_max_depth', 10.)

        #### camera intrinsics
        #             (u)   (x)
        #    d * K^-1 (v) = (y)
        #             (1)   (z)

        ## renderer for visualization
        R = [[[1.,0.,0.],
              [0.,1.,0.],
              [0.,0.,1.]]]
        R = torch.FloatTensor(R).to(self.device)
        t = torch.zeros(1,3, dtype=torch.float32).to(self.device)
        fx = (self.image_size-1)/2/(math.tan(self.fov/2 *math.pi/180))
        fy = (self.image_size-1)/2/(math.tan(self.fov/2 *math.pi/180))
        cx = (self.image_size-1)/2
        cy = (self.image_size-1)/2
        K = [[fx, 0., cx],
             [0., fy, cy],
             [0., 0., 1.]]
        K = torch.tensor(K, dtype = torch.float32)
        self.inv_K = torch.linalg.inv(K).unsqueeze(0).to(self.device)
        self.K = K.unsqueeze(0).to(self.device)
        self.renderer = Renderer(camera_mode='projection',
                                    light_intensity_ambient=1.0,
                                    light_intensity_directional=0.,
                                    K=self.K, R=R, t=t,
                                    near=self.renderer_min_depth, far=self.renderer_max_depth,
                                    image_size=self.image_size, orig_size=self.image_size,
                                    fill_back=True,
                                    background_color=[1,1,1])

    def set_transform_matrices(self, view):
        self.rot_mat, self.trans_xyz = get_transform_matrices(view)

    def rotate_pts(self, pts, rot_mat):
        centroid = torch.FloatTensor([0.,0.,self.rot_center_depth]).to(pts.device).view(1,1,3)
        pts = pts - centroid  # move to centroid
        pts = pts.matmul(rot_mat.transpose(2,1))  # rotate
        pts = pts + centroid  # move back
        return pts

    def translate_pts(self, pts, trans_xyz):
        return pts + trans_xyz

    def depth_to_3d_grid(self, depth):
        b, h, w = depth.shape
        grid_2d = get_grid(b, h, w, normalize=False).to(depth.device)  # Nxhxwx2
        depth = depth.unsqueeze(-1)
        grid_3d = torch.cat((grid_2d, torch.ones_like(depth)), dim=3)
        grid_3d = grid_3d.matmul(self.inv_K.to(depth.device).transpose(2,1)) * depth
        return grid_3d

    def grid_3d_to_2d(self, grid_3d):
        b, h, w, _ = grid_3d.shape
        grid_2d = grid_3d / grid_3d[...,2:]
        grid_2d = grid_2d.matmul(self.K.to(grid_3d.device).transpose(2,1))[:,:,:,:2]
        WH = torch.FloatTensor([w-1, h-1]).to(grid_3d.device).view(1,1,1,2)
        grid_2d = grid_2d / WH *2.-1.  # normalize to -1~1
        return grid_2d

    def get_warped_3d_grid(self, depth):
        b, h, w = depth.shape
        grid_3d = self.depth_to_3d_grid(depth).reshape(b,-1,3)
        grid_3d = self.rotate_pts(grid_3d, self.rot_mat)
        grid_3d = self.translate_pts(grid_3d, self.trans_xyz)
        return grid_3d.reshape(b,h,w,3) # return 3d vertices

    def get_inv_warped_3d_grid(self, depth):
        b, h, w = depth.shape
        grid_3d = self.depth_to_3d_grid(depth).reshape(b,-1,3)
        grid_3d = self.translate_pts(grid_3d, -self.trans_xyz)
        grid_3d = self.rotate_pts(grid_3d, self.rot_mat.transpose(2,1))
        return grid_3d.reshape(b,h,w,3) # return 3d vertices

    def get_warped_2d_grid(self, depth):
        b, h, w = depth.shape
        grid_3d = self.get_warped_3d_grid(depth)
        grid_2d = self.grid_3d_to_2d(grid_3d)
        return grid_2d

    def get_inv_warped_2d_grid(self, depth):
        b, h, w = depth.shape
        grid_3d = self.get_inv_warped_3d_grid(depth)
        grid_2d = self.grid_3d_to_2d(grid_3d)
        return grid_2d

    def warp_canon_depth(self, canon_depth):
        b, h, w = canon_depth.shape
        grid_3d = self.get_warped_3d_grid(canon_depth).reshape(b,-1,3)
        faces = get_face_idx(b, h, w).to(canon_depth.device)
        warped_depth = self.renderer.render_depth(grid_3d, faces)

        # allow some margin out of valid range
        margin = (self.max_depth - self.min_depth) /2
        warped_depth = warped_depth.clamp(min=self.min_depth-margin, max=self.max_depth+margin)
        return warped_depth


def mm_normalize(x, min=0, max=1):
    x_min = x.min()
    x_max = x.max()
    x_range = x_max - x_min
    x_z = (x - x_min) / x_range
    x_out = x_z * (max - min) + min
    return x_out


def rand_range(size, min, max):
    return torch.rand(size)*(max-min)+min


def rand_posneg_range(size, min, max):
    i = (torch.rand(size) > 0.5).type(torch.float)*2.-1.
    return i*rand_range(size, min, max)


def get_grid(b, H, W, normalize=True):
    if normalize:
        h_range = torch.linspace(-1,1,H)
        w_range = torch.linspace(-1,1,W)
    else:
        h_range = torch.arange(0,H)
        w_range = torch.arange(0,W)
    grid = torch.stack(torch.meshgrid([h_range, w_range]), -1).repeat(b,1,1,1).flip(3).float() # flip h,w to x,y
    return grid


def get_rotation_matrix(tx, ty, tz):
    m_x = torch.zeros((len(tx), 3, 3)).to(tx.device)
    m_y = torch.zeros((len(tx), 3, 3)).to(tx.device)
    m_z = torch.zeros((len(tx), 3, 3)).to(tx.device)

    m_x[:, 1, 1], m_x[:, 1, 2] = tx.cos(), -tx.sin()
    m_x[:, 2, 1], m_x[:, 2, 2] = tx.sin(), tx.cos()
    m_x[:, 0, 0] = 1

    m_y[:, 0, 0], m_y[:, 0, 2] = ty.cos(), ty.sin()
    m_y[:, 2, 0], m_y[:, 2, 2] = -ty.sin(), ty.cos()
    m_y[:, 1, 1] = 1

    m_z[:, 0, 0], m_z[:, 0, 1] = tz.cos(), -tz.sin()
    m_z[:, 1, 0], m_z[:, 1, 1] = tz.sin(), tz.cos()
    m_z[:, 2, 2] = 1
    return torch.matmul(m_z, torch.matmul(m_y, m_x))


def get_transform_matrices(view):
    b = view.size(0)
    if view.size(1) == 6:
        rx = view[:,0]
        ry = view[:,1]
        rz = view[:,2]
        trans_xyz = view[:,3:].reshape(b,1,3)
    elif view.size(1) == 5:
        rx = view[:,0]
        ry = view[:,1]
        rz = view[:,2]
        delta_xy = view[:,3:].reshape(b,1,2)
        trans_xyz = torch.cat([delta_xy, torch.zeros(b,1,1).to(view.device)], 2)
    elif view.size(1) == 3:
        rx = view[:,0]
        ry = view[:,1]
        rz = view[:,2]
        trans_xyz = torch.zeros(b,1,3).to(view.device)
    rot_mat = get_rotation_matrix(rx, ry, rz)
    return rot_mat, trans_xyz


def get_face_idx(b, h, w):
    idx_map = torch.arange(h*w).reshape(h,w)
    faces1 = torch.stack([idx_map[:h-1,:w-1], idx_map[1:,:w-1], idx_map[:h-1,1:]], -1).reshape(-1,3)
    faces2 = torch.stack([idx_map[:h-1,1:], idx_map[1:,:w-1], idx_map[1:,1:]], -1).reshape(-1,3)
    return torch.cat([faces1,faces2], 0).repeat(b,1,1).int()



'''
------------------------------- Authro's renderer -------------------------------
'''

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

def test_depth_to_normal_v2():
    inp = torch.randn(17, 1, 64, 64)
    res2 = depth_to_normal_v2(None, inp)
    print(res2.shape)
    print(res2[0,0,:,:])

    for i in range(17):
        out = depth_to_normal(None, inp[i])
        if i == 0:
            res_org = out
        else:
            res_org = torch.cat([res_org, out], dim=0)
    
    print(res_org.shape)

    print("res2_mean: ", torch.mean(res2, dim=(0,1,2,3)))
    print("res_org_mean: ", torch.mean(res_org, dim=(0,1,2,3)))
    print("difference: ", torch.mean(res_org-res2, dim=(0,1,2,3)))
    print((res2-res_org)[0,0,:,:])


def depth_to_normal_v2(self, depth_map):
    '''
    input:
    - depth_map : B x 1 x W x H
    output:
    - normal_map : B x 3 x W x H
    '''
    B, _,  W, H = depth_map.shape 
    x_grid = torch.linspace(0, H-1, H, dtype = torch.float32)
    y_grid = torch.linspace(0, W-1, W, dtype = torch.float32)

    xx, yy = torch.meshgrid(x_grid, y_grid, indexing = 'ij')
    xx = xx.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)
    yy = yy.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)

    depth_x_s = torch.zeros_like(depth_map)
    depth_y_s = torch.zeros_like(depth_map)
    depth_x_s[:,:,1:W,:] = depth_map[:,:,0:W-1,:]
    depth_y_s[:,:,:,1:H] = depth_map[:,:,:,0:H-1]

    v1 = torch.cat([xx, (yy-1), depth_y_s], dim = 1)
    v2 = torch.cat([(xx-1), yy, depth_x_s], dim = 1)
    c = torch.cat([xx,yy,depth_map], dim = 1)

    d = torch.cross(v2-c, v1-c, dim = 1)
    n = d / torch.sqrt(torch.sum(d**2, dim = 1, keepdim=True))

    return n


def depth_to_normal(self, depth_map):
    '''
    - input: depth_map (size: torch.Size([1, 64, 64]))
    - output: normal_map (size: torch.Size([1, 64, 64])
    '''
    d, w, h = depth_map.shape
    normal_map = torch.zeros_like(depth_map).repeat(3, 1, 1)

    # calculate normal map from depth map
    for i in range(1, w - 1):
        for j in range(1, h - 1):
            v1 = torch.Tensor([i, j - 1, depth_map[0][i][j - 1]])
            v2 = torch.Tensor([i - 1, j, depth_map[0][i - 1][j]])
            c = torch.Tensor([i, j, depth_map[0][i][j]])

            d = torch.cross(v2 - c, v1 - c)
            n = d / torch.sqrt(torch.sum(d ** 2))
            normal_map[:, i, j] = n

    return normal_map.unsqueeze(0)


def test_gen_grid_0513():
    path1 = "./test1.jpg"   
    path2 = "./test2.jpg"
    imgs1 = get_torch_image(13, 64, 64, path1).cuda()
    imgs2 = get_torch_image(13, 64, 64, path2).cpu()

    grid1 = gen_grid(imgs1)
    grid2 = gen_grid(imgs2)

    print("grid 1 shape:", grid1.shape, "grid 1 device", grid1.device)
    print("grid 2 shape:", grid2.shape, "grid 2 device", grid2.device)
    print(grid2[0,:,0:10,0:10])


def test_matmul_0513_1():
    path1 = "./test1.jpg"   
    path2 = "./test2.jpg"
    imgs1 = get_torch_image(13, 64, 64, path1)
    imgs2 = get_torch_image(13, 64, 64, path2)

    if True:
        # test with mixed batch
        imgs1 = torch.cat([imgs1, imgs2], dim=0)
        
    K = torch.eye(3)

    print("imgs1 shape: ", imgs1.shape)
    print("K shape: ",K.shape)

    try:
        res1 = torch.matmul(K, imgs1)
        print(res1.shape)
    except RuntimeError as e:
        print("trial 1", e)

    try:
        K_t = K.unsqueeze(0).repeat(13,1,1)
        res2 = torch.matmul(K_t, imgs1)
        print(res2.shape)
    except RuntimeError as e:
        print("trial 2", e)

    try:
        K_t = K.unsqueeze(0).repeat(13,1,1).unsqueeze(-1)
        res3 = torch.matmul(K_t, imgs1)
        print(res3.shape)
    except RuntimeError as e:
        print("trial 3", e)

    try:        ######## -> selected ! only function that works porperly here!
        imgs1_t = imgs1.reshape(imgs1.shape[0],imgs1.shape[1],-1)
        res4 = torch.matmul(K, imgs1_t)
        res4 = res4.reshape(imgs1.shape[0],imgs1.shape[1],imgs1.shape[2],imgs1.shape[3])
        print(res4.shape)
        diff = res4-imgs1
        print("trial 4 diff mean: ", torch.mean(diff), "diff std: ", torch.std(diff))
    except RuntimeError as e:
        print("trial 4", e)

    try:
        imgs1_t = imgs1.reshape(imgs1.shape[0],imgs1.shape[1],-1).permute(1,0,2)
        res4 = torch.matmul(K, imgs1_t).permute(1,0,2)
        res4 = res4.reshape(imgs1.shape[0],imgs1.shape[1],imgs1.shape[2],imgs1.shape[3])
        print(res4.shape)
        diff = res4-imgs1
        print("trial 5 diff mean: ", torch.mean(diff), "diff std: ", torch.std(diff))
    except RuntimeError as e:
        print("trial 5", e)


def test_matmul_0513_2():       # to test (B x 3 x W x H) x (3 x 3)
    path1 = "./test1.jpg"   
    path2 = "./test2.jpg"
    imgs1 = get_torch_image(13, 64, 64, path1)
    imgs2 = get_torch_image(13, 64, 64, path2)

    if True:
        # test with mixed batch
        imgs1 = torch.cat([imgs1, imgs2], dim=0)
        
    K = torch.eye(3)

    print("imgs1 shape: ", imgs1.shape)
    print("K shape: ",K.shape)

    try:        ######## -> selected ! only function that works porperly here!
        imgs1_t = imgs1.reshape(imgs1.shape[0],imgs1.shape[1],-1).permute(0,2,1)
        res4 = torch.matmul(imgs1_t, K).permute(0,2,1)
        res4 = res4.reshape(imgs1.shape[0],imgs1.shape[1],imgs1.shape[2],imgs1.shape[3])
        print(res4.shape)
        diff = res4-imgs1
        print("trial 4 diff mean: ", torch.mean(diff), "diff std: ", torch.std(diff))
    except RuntimeError as e:
        print("trial 4", e)


def test_safe_matmul_0513():
    path1 = "./test1.jpg"   
    path2 = "./test2.jpg"
    imgs1 = get_torch_image(13, 64, 64, path1)
    imgs2 = get_torch_image(13, 64, 64, path2)

    if True:
        # test with mixed batch
        imgs1 = torch.cat([imgs1, imgs2], dim=0)
        
    K = torch.eye(3)

    res1 = safe_matmul(K, imgs1)
    diff = res1-imgs1
    print("trial 1 diff mean: ", torch.mean(diff), "diff std: ", torch.std(diff))

    res2 = safe_matmul(imgs1, K)
    diff = res2-imgs1
    print("trial 2 diff mean: ", torch.mean(diff), "diff std: ", torch.std(diff))


def test_0514(test_util = True):
    os.makedirs("./debug",exist_ok=True)
    DATA_PATH = "./human_face"
    face_list = [
        '001_face',
        '002_face',
        '003_face',
        '004_face',
        '005_face'
    ]
    b_size = 5
    interm_values = []
    for f_name in face_list:
        dir_name = os.path.join(DATA_PATH, f_name)
        res = test_data_loader(dir_name)
        interm_values.append(res)

        if False:
            # check meaning of "ori" of data
            imgdir_name = os.path.join("./debug", f_name)
            os.makedirs(imgdir_name, exist_ok=True)
            img_ori = tensor_to_image(res['canon_depth_ori'])
            img_raw = tensor_to_image(res['canon_depth_raw'])
            img_alb = tensor_to_image(res['canon_albedo'])
            img_can = tensor_to_image((res['canon_img']+1)/2)
            img_ori.save(os.path.join(imgdir_name, "depth_ori.png"))
            img_raw.save(os.path.join(imgdir_name, "depth_raw.png"))
            img_alb.save(os.path.join(imgdir_name, "canon_albedo.png"))
            img_can.save(os.path.join(imgdir_name, "canon_img.png"))
            assert(0)
    
    # make_batch with given data
    b_depth = res['canon_depth_ori'].unsqueeze(0)     # B 1 W H
    b_albedo = res['canon_albedo']                    # B 3 W H
    b_view = res['view']
    b_canon_img = res['canon_img']

    # to check utils part
    b_light = res['light']
    b_shading = res['canon_shading']
    b_normal = res['canon_normal']

    #iteratively build batch
    for i in range(4):
        res = interm_values[i]
        b_depth = torch.cat([b_depth, res['canon_depth_ori'].unsqueeze(0)], dim=0)
        b_albedo = torch.cat([b_albedo, res['canon_albedo'] ], dim=0)
        b_view = torch.cat([b_view, res['view'] ], dim=0)
        b_canon_img = torch.cat([b_canon_img, res['canon_img'] ], dim=0)

        b_light = torch.cat([b_light, res['light'] ], dim=0)
        b_shading = torch.cat([b_shading, res['canon_shading'] ], dim=0)
        b_normal = torch.cat([b_normal, res['canon_normal'] ], dim=0)

    print(b_light)
    b_light = b_light.unsqueeze(-1).unsqueeze(-1)

    print("about raw depth")

    if test_util:
        print("light min:", b_light.min(), "light max:", b_light.max())
        test_utils(b_depth, b_light, b_albedo, b_shading, b_normal, b_canon_img)

    else:
        im_a, dep_a = authors_render(b_depth, b_canon_img, b_view)
        im_t, dep_t = test_render(b_depth, b_canon_img, b_view)

        im_diff = torch.abs(im_a - im_t)
        dep_diff = torch.abs(dep_a - dep_t)

        print("image diff mean: ", im_diff.mean(), ", min: ", im_diff.min(),", max: ", im_diff.max())
        print("depth diff mean: ", dep_diff.mean(), ", min: ", dep_diff.min(),", max: ", dep_diff.max())

        compare_plot(
            './debug/renderer.png',
            im_diff.detach().cpu(), 
            dep_diff.detach().cpu(), 
            title1='img diff', 
            title2='depth diff'
        )
        
        return b_view
        

def test_utils(depth, light, albedo, shading, normal, canon):
    ImgForm = ImageFormation(device = torch.device("cpu"))
    
    RDR = RenderPipeline(b_size = depth.shape[0])
    #print(light)

    light = light.squeeze()
    org_light = deepcopy(light)
    org_light = (org_light+1.)/2.
    #light[:,0:2] = light[:,0:2]*2. -1.

    print(albedo.min(), albedo.max())
    print(depth.shape)
    print(normal.shape)
    au_my_normal = RDR.depth2normal_author(depth.cuda())
    my_normal = ImgForm.depth_to_normal(depth)
    print(my_normal.shape)
    my_shading = ImgForm.normal_to_shading(normal.permute(0,3,1,2), light.squeeze())
    my_canon = ImgForm.alb_to_canon(albedo, my_shading)

    print(light.squeeze())
    au_my_shading = ImgForm.normal_to_shading(au_my_normal.cpu(), light.squeeze())
    au_my_canon = ImgForm.alb_to_canon((albedo+1.)/2., au_my_shading)


    print("org au my shading: ", au_my_shading.shape, ", min: ", au_my_shading.min(),", max: ", my_shading.max())
    # check whether it's changed by light condition
    # print(org_light)
    au_my_shading_1 = au_my_shading.detach().cpu()
    au_my_shading_2 = au_my_shading_1 - org_light[:,0].view(-1,1,1,1)
    au_my_shading_3 = au_my_shading_2 / org_light[:,1].view(-1,1,1,1)
    au_my_shading = au_my_shading_3
    #compare_plot('./debug/normal.png', my_normal, au_my_normal.detach().cpu())
    compare_plot(
        './debug/depth.png',
        au_my_shading.detach(), 
        shading, 
        title1='authorss mine', 
        title2='original code'
    )
    compare_plot(
        './debug/depth.png',
        au_my_canon.detach(), 
        canon, 
        title1='authorss mine', 
        title2='original code'
    )
    compare_plot(
        './debug/depth.png',
        au_my_normal.detach().cpu(), 
        normal.permute(0,3,1,2), 
        title1='authorss mine', 
        title2='original code'
    )
    
    # compare normal
    print("my normal: ", my_normal.shape, ", min: ", my_normal.min(),", max: ", my_normal.max())
    print("org normal: ", normal.shape, ", min: ", normal.min(),", max: ", normal.max())
    compare_plot('./debug/normal.png', my_normal, normal.permute(0,3,1,2))

    # compare shading
    print("au my shading: ", au_my_shading.shape, ", min: ", au_my_shading.min(),", max: ", my_shading.max())
    print("org shading: ", shading.shape, ", min: ", shading.min(),", max: ", shading.max())
    compare_plot('./debug/shadings.png', my_shading, shading)

    # compare canon
    print("my canon: ", au_my_canon.shape, ", min: ", au_my_canon.min(),", max: ", au_my_canon.max())
    print("org canon: ", canon.shape, ", min: ", canon.min(),", max: ", canon.max())
    compare_plot('./debug/canon.png', au_my_canon, canon, albedo)

    
    shade_diff = torch.abs(au_my_shading - shading)
    normal_diff = torch.abs(au_my_normal.detach().cpu() - normal.permute(0,3,1,2))

    print("shading diff mean: ", shade_diff.mean(), ", min: ", shade_diff.min(),", max: ", shade_diff.max())
    print("normal diff mean: ", normal_diff.mean(), ", min: ", normal_diff.min(),", max: ", normal_diff.max())

    compare_plot(
        './debug/renderer.png',
        shade_diff.detach().cpu(), 
        normal_diff.detach().cpu(), 
        title1='shading diff', 
        title2='normal diff'
    )

def test_render(depth, img, a_view):
    view = torch.zeros_like(a_view).to(a_view.device)
    view[:,0:3] = a_view[:,0:3]*180./math.pi/60.
    view[:,3:] = a_view[:,3:]*10.


    RDR = RenderPipeline(b_size = depth.shape[0])
    org_img, org_depth = RDR(
        canon_depth = depth.cuda(),
        canon_img = img.cuda(),
        views = view.cuda()
    )
    org_img = org_img * (org_depth != 1.2)
    print("org image: ", org_img.shape, ", min: ", org_img.min(),", max: ", org_img.max())
    print("org depth: ", org_depth.shape, ", min: ", org_depth.min(),", max: ", org_depth.max())
    compare_plot(
        './debug/depth.png',
        img, 
        depth, 
        title1='canon_img', 
        title2='depth'
    )
    compare_plot(
        './debug/renderer.png',
        org_img.detach().cpu(), 
        org_depth.detach().cpu(), 
        title1='recon_img', 
        title2='depth'
    )

    return org_img, org_depth
    
    #plot_3d(depth[0,0,:,:].detach().cpu(), 4)
    #plot_3d(org_depth[0,0,:,:].detach().cpu(), 5)
    


def authors_render(depth, img, view):
    depth = depth.cuda()
    img = img.cuda()
    view = view.cuda()


    ## reconstruct input view
    renderer = ARenderer(cfgs)
    renderer.set_transform_matrices(view.squeeze())
    recon_depth = renderer.warp_canon_depth(depth.squeeze())
    grid_2d_from_canon = renderer.get_inv_warped_2d_grid(recon_depth)
    recon_im = nn.functional.grid_sample(img, grid_2d_from_canon, mode='bilinear', align_corners = True)

    print(recon_depth.shape)
    print(recon_im.shape)

    recon_depth = recon_depth.unsqueeze(1)

    org_img = recon_im * (recon_depth != 1.2)
    print("org image: ", org_img.shape, ", min: ", org_img.min(),", max: ", org_img.max())
    print("org depth: ", recon_depth.shape, ", min: ", recon_depth.min(),", max: ", recon_depth.max())

    compare_plot(
        './debug/depth.png',
        img.cpu(), 
        depth.cpu(), 
        title1='canon_img', 
        title2='depth'
    )
    compare_plot(
        './debug/renderer.png',
        org_img.detach().cpu(), 
        recon_depth.detach().cpu(), 
        title1='recon_img', 
        title2='depth'
    )

    
    return org_img, recon_depth


def compare_plot(fname, imgs1, imgs2, imgs3=None, title1='mine', title2='authors', title3='albedo'):
    '''
    plot images, in single PNG file.
    Row : batch, col : img variation

    imgs1 : mine
    imgs2 : original (author's)
    '''
    B = imgs1.shape[0]

    imgs1 = to_zeroone(imgs1)
    imgs2 = to_zeroone(imgs2)

    if imgs3 is not None:
        imgs3 = to_zeroone(imgs3)

    if imgs3 is None:
        fig, axs = plt.subplots(2,B)
    else:
        fig, axs = plt.subplots(3,B)
    for i in range(B):
        img1 = imgs1[i].permute(1,2,0)
        img2 = imgs2[i].permute(1,2,0)

        axs[0, i].imshow(img1)
        axs[0, i].set_title(title1)
        axs[1, i].imshow(img2)
        axs[1, i].set_title(title2)

        if imgs3 is not None:
            img3 = imgs3[i].permute(1,2,0)
            axs[2, i].imshow(img3)
            axs[2, i].set_title(title3)

    fig.savefig(fname)


def to_zeroone(tensor):
    min = tensor.min()
    max = tensor.max()
    return (tensor - min)/(max-min+EPS)


def test_data_loader(path):
    '''
    canon_albedo        shape:  torch.Size([1, 3, 64, 64])  -> (-1 ~ 1)
    canon_depth_ori     shape:  torch.Size([1, 64, 64])     -> clamped data (0.9~1.1)
    canon_depth_raw     shape:  torch.Size([1, 64, 64])     -> useless (-1~10 values)
    light               shape:  torch.Size([1, 4])          
    view                shape:  torch.Size([1, 6])
    canon_img           shape:  torch.Size([1, 3, 64, 64])  -> (-1 ~ 1)
    canon_shading       shape:  torch.Size([1, 1, 64, 64])
    canon_normal        shape:  torch.Size([1, 64, 64, 3])

    It seems like is saved the "raw output" of the network.
    I need to modify it to use it.
    '''

    interm_values = {
        'canon_albedo':torch.load(os.path.join(path, 'canon_albedo_ori.pt')),
        'canon_depth_ori':torch.load(os.path.join(path, 'canon_depth_ori.pt')),
        'canon_depth_raw':torch.load(os.path.join(path, 'canon_depth_raw.pt')),
        'light':torch.load(os.path.join(path, 'canon_light.pt')),
        'view':torch.load(os.path.join(path, 'view.pt')),
        'canon_img':torch.load(os.path.join(path, 'canon_im_ori.pt')),
        'canon_shading':torch.load(os.path.join(path, 'canon_diffuse_shading_ori.pt')),
        'canon_normal':torch.load(os.path.join(path, 'canon_normal_ori.pt')),
    }

    if False:
        # print tensor size
        for key in list(interm_values.keys()):
            print(key, " shape: ", interm_values[key].shape)
    
    return interm_values



def tensor_to_image(tensor):
    if tensor.shape[1] == 3:
        tensor = tensor.permute(0,2,3,1)
    tensor_np = (tensor.squeeze()*255).to(dtype=torch.uint8).numpy()
    img = Image.fromarray(tensor_np)
    return img





def plot_3d(depth, fig_ind):
    '''
    plot depth surface in 3D
    '''
    W, H = depth.squeeze().shape

    lin_X = torch.linspace(0, W-1, W)
    lin_Y = torch.linspace(0, H-1, H)
    xx, yy = torch.meshgrid(lin_X, lin_Y, indexing = 'ij', )

    fig = plt.figure(fig_ind)
    ax = plt.axes(projection='3d')
    ax.contour3D(xx, yy, depth.squeeze(), 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


        







if __name__ == '__main__':
    test_0514(test_util=True)
    test_0514(test_util=False)


