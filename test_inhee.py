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

    b_light = b_light.unsqueeze(-1).unsqueeze(-1)


    if test_util:
        test_utils(b_depth, b_light, b_albedo, b_shading, b_normal, b_canon_img)

    else:
        test_render(b_depth, b_canon_img, b_view)

def test_utils(depth, light, albedo, shading, normal, canon):
    ImgForm = ImageFormation(device = torch.device("cpu"))
    
    RDR = RenderPipeline(b_size = depth.shape[0])
    print(depth.shape)
    print(normal.shape)
    au_my_normal = RDR.depth2normal_author(depth.cuda())
    my_normal = ImgForm.depth_to_normal(depth)
    print(my_normal.shape)
    my_shading = ImgForm.normal_to_shading(normal.permute(0,3,1,2), light.squeeze())
    my_canon = ImgForm.alb_to_canon(albedo, my_shading)


    compare_plot('./debug/normal.png', my_normal, au_my_normal.detach().cpu())

    
    # compare normal
    print("my normal: ", my_normal.shape, ", min: ", my_normal.min(),", max: ", my_normal.max())
    print("org normal: ", normal.shape, ", min: ", normal.min(),", max: ", normal.max())
    compare_plot('./debug/normal.png', my_normal, normal.permute(0,3,1,2))

    # compare shading
    print("my shading: ", my_shading.shape, ", min: ", my_shading.min(),", max: ", my_shading.max())
    print("org shading: ", shading.shape, ", min: ", shading.min(),", max: ", shading.max())
    compare_plot('./debug/shadings.png', my_shading, shading)

    # compare canon
    print("my canon: ", my_canon.shape, ", min: ", my_canon.min(),", max: ", my_canon.max())
    print("org canon: ", canon.shape, ", min: ", canon.min(),", max: ", canon.max())
    compare_plot('./debug/canon.png', my_canon, canon, albedo)

    


def test_render(depth, img, view):
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
    
    #plot_3d(depth[0,0,:,:].detach().cpu(), 4)
    #plot_3d(org_depth[0,0,:,:].detach().cpu(), 5)
    




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
    #test_0514(test_util=False)


