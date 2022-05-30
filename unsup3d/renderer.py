import torch
import torch.nn as nn
import math
from neural_renderer import Renderer
from unsup3d.utils import *

BATCH_SIZE = 16
EPS = 1e-7

DEPTH_ALLOW_MARGIN = 0.2        # originally, it's 0.1
IS_DEBUG = False
USE_WIDER_DEPTH = False

class RenderPipeline(nn.Module):
    def __init__(self, device = torch.device("cuda"), b_size = BATCH_SIZE, args = None):
        '''
        this part should be received from argument "args"
        We need to update it! (05/13 inhee) 
        '''
        super(RenderPipeline, self).__init__()
        W = 64
        H = 64
        B = b_size
        theta_fov = 10  # 10 degree, not rad
        self.W = W
        self.H = H
        self.device = device
        self.min_depth = 0.9    # from Table.6 of paper
        self.max_depth = 1.1    # from Table.6 of paper

        center_depth = (self.min_depth + self.max_depth)/2     # use to centralize before rotation
        center_depth = torch.tensor([[[[center_depth]]]], dtype = torch.float32) # 1 x 1 x 1 x 1
        self.center_depth = torch.cat([
            torch.zeros(1,2,1,1, dtype = torch.float32),
            center_depth], dim = 1).to(self.device)            # 1 x 3 x 1 x 1

        self.render_min_depth = 0.1         # allow larger depth range on renderer, than clamp
        self.render_max_depth = 10.0        # allow larger depth range on renderer, than clamp


        # define inverse function
        c_u = (W-1)/2
        c_v = (H-1)/2
        f = (W-1)/(2*math.tan(deg2rad(theta_fov)/2))

        self.K = torch.tensor([
            [f, 0, c_u],
            [0, f, c_v],
            [0, 0, 1]], dtype = torch.float32)              
        self.K_inv = torch.linalg.inv(self.K)               # 3x3 matrix
        self.K = self.K.unsqueeze(0).to(self.device)
        self.K_inv = self.K_inv.unsqueeze(0).to(self.device)


        # define neural renderer
        # Here, we give camer information by K, I omitted camera relevant args (05/13)
        T = torch.zeros(1,3,dtype = torch.float32).to(self.device)  # zero translation for renderer
        R = torch.eye(3,dtype = torch.float32).to(self.device)      # zero rotation for renderer
        R = R.unsqueeze(0)

        self.renderer = Renderer(
            image_size=W,                       # output image size. It can only handle square
            anti_aliasing=True,                 
            background_color=[1,1,1],           # background color, [0,0,0] : black, [1,1,1] : white
            fill_back=True,                     # whether fill background    
            camera_mode='projection',           # use 'projection', if you want to use others like 'look' or 'look_at',
                                                # you should add camere relevant args
            K=self.K,                           # vertices projecting matrix
            R=R,                                # object rotating matrix (can also apply in method-call)
            t=T,                                # object translating matrix (can also apply in method-call)
            #dist_coeffs=None,                   # vector of distortion coefficients, default : None 
            orig_size=W,                        # original size of image captured by the camera 
            near = self.render_min_depth,              # minmum depth
            far = self.render_max_depth,               # maximum depth
            light_intensity_ambient=1.0, 
            light_intensity_directional=0.,
            #light_color_ambient=[1,1,1], 
            #light_color_directional=[1,1,1],
            #light_direction=[0,1,0]
        )

        # define faces here
        # it can be reused, unless the input shape or batch_size changes
        # we should drop last-batch on dataloader!!!!!  ----------------probably erroneous-----------------> (05/15 inhee)
        self.faces = get_faces(B, W, H).to(self.device)
        

    def canon_depth_to_3d(self, canon_depth):
        '''
        input:
        - canon_depth : depth map (B x 1 x W x H)
        output:
        - canon_pc : canoncial point cloud (B x 3 x W x H)

        it changes 1D depth map to 3D point cloud
        output = depth * inv_K * (u,v,1)
        (05/13 inhee)
        '''
        grid = gen_grid(canon_depth)      # B 2 W H
        grid_3d = torch.cat(
            [grid, torch.ones_like(canon_depth, dtype = torch.float32).to(canon_depth.device)], 
            dim=1)                  # B 3 W H
        
        canon_pc = safe_matmul(grid_3d, self.K_inv.transpose(2,1)) # B 3 W H
        canon_pc = canon_pc * canon_depth                      # B 3 W H

        return canon_pc


    def canon_3d_to_org_3d(self, canon_pc, rotates, trans):
        '''
        input:
        - canon_pc : canonical point cloud (B x 3 x W x H)
        - rotate : rotation angle, (-60 ~ 60 degree) (B x 3)
        - trans : translation value, (-0.1 ~ 0.1) (B x 3)  
        output:
        - org_pc : original point cloud (B x 3 x W x H)

        it applies camera viewpoint, by rotating & translating
        output = Rot ( input ) + Trans

        (05/13 inhee)
        '''
        self.R = get_rot_mat(
            alpha = rotates[:,2:3],
            beta = rotates[:,0:1],
            gamma = rotates[:,1:2]
        )
        self.T = trans.unsqueeze(-1).unsqueeze(-1)  # B 3 1 1

        # first centeralize & rotate & relocate
        c_canon_pc = canon_pc - self.center_depth    # B 3 W H
        org_pc = safe_matmul(c_canon_pc, self.R.transpose(2,1))     # B 3 W H
        org_pc = org_pc + self.center_depth
        org_pc = org_pc + self.T

        return org_pc


    def org_3d_to_org_depth(self, org_pc):
        '''
        input:
        - org_pc : original point cloud (B x 3 x W x H)
        output:
        - org_depth : original depth map (B x 1 x W x H)

        it render pointcloud with Renderer.

        <special point>
        it allows margin of depth
        (basically, the depth map should be ~ (0.9, 1.1))
        However, it can avoid fitting on boundary value & it's just limitation on 
        canonical depth map.
        -> so allow margin +- 0.1 so it has value ~ (0.8, 1.2)
        -> it's only mentioned in authors' code

        (05/14 inhee)
        '''
        if len(org_pc.shape) != 4:
            print("wrong input shape")
            print("current org_pc shape: ", org_pc.shape)
            assert(0)
        
        B, _, W, H = org_pc.shape
        device = org_pc.device
        
        # get vertices
        vertices = org_pc.reshape(B, 3, -1)    # B x 3 x WH
        vertices = vertices.permute(0, 2, 1)   # B x WH x 3
       
        org_depth = self.renderer.render_depth(
            vertices = vertices,                     # B x N_vertices(WH) x 3
            faces = self.faces,                      # B x N_faces(2(W-1)(H-1)) x 3
        )

        # allow extra margin
        if USE_WIDER_DEPTH:
            margin = (self.max_depth - self.min_depth)
        else:
            margin = (self.max_depth - self.min_depth) / 2

        org_depth = org_depth.clamp(min = self.min_depth - margin, max = self.max_depth + margin)

        return org_depth.unsqueeze(1)


    def get_warp_grid(self, org_depth):
        '''
        input:
        - org_depth : original depth map (B x 1 x W x H)
        output:
        - warp_grid : grid information that which points of canoncial image is used
                     for generating orginal grid (B x 2 x W x H)

        By projecting each 2D image into 3D space, inverse Rotate and inverse translate,
        and finally reprojecting with known camera setting (K), we can get (u,v,1) for each
        original depth map's pixel.
        easily, (u,v,1) of D_org(x,y) of output means that D_org(x,y) is determined by sampling
        depth of (u,v) of canoncial depth map.
        It can be also applied to canonical image if we know that warping relation.
        This function is for extracting those warping function. 
        (05/13 inhee)
        '''
        grid = gen_grid(org_depth)      # B 2 W H
        grid_3d = torch.cat(
            [grid, torch.ones_like(org_depth, dtype = torch.float32).to(org_depth.device)], 
            dim=1)                      # B 3 W H
        org_pc = safe_matmul(grid_3d, self.K_inv.transpose(2,1))
        org_pc = org_pc * org_depth
        
        canon_pc = org_pc - self.T                                  # B 3 W H

        # first centeralize & rotate & relocate
        c_canon_pc = canon_pc - self.center_depth
        canon_pc = safe_matmul(c_canon_pc, self.R)   # B 3 W H, inv rotation == transpose
        canon_pc = canon_pc + self.center_depth


        canon_pc = canon_pc / (canon_pc[:,2:3,:,:] + EPS)
        canon_pc = safe_matmul(canon_pc,self.K.transpose(2,1))
        
        warp_grid = canon_pc[:,0:2,:,:]                             # B 2 W H
        
        # change it to have value range -1 ~ 1
        # The grid would hold value 0~W-1, 0~H-1
        normalize = torch.tensor([self.W-1, self.H-1], dtype = torch.float32).to(self.device)
        normalize = normalize.view(1,2,1,1) / 2.0
        warp_grid = warp_grid/normalize - 1.0

        return warp_grid



    def get_org_image(self, warp_grid, canon_img):
        '''
        input:
        - warp_grid : warping matrix that sample canonical to make original view
                     (B x 2 x W x H)
        - canon_img : canoncial image (B x 3 x W x H)
        output:
        - org_img : original image (B x 3 x W x H)

        Simply get original image, by grid_sample function
        (05/13 inhee)
        '''

        org_img = nn.functional.grid_sample(
            input = canon_img,                          # B 3 W H
            grid = warp_grid.permute(0,2,3,1),          # B W H 2
            mode = 'bilinear',
            padding_mode = 'zeros',
            align_corners = True
        )
        

        return org_img


    def forward(self, canon_depth, canon_img, views):
        '''
        input:
        - canon_depth : canonical depth map from DephtNet (B x 1 x W x H)
        - canon_img : canonical image from pipeline, (B x 3 x W x H)
        - views : raw output of ViewNet, (-0.1 ~ 0.1) (+60, -60) (B x 6)  
        output:
        - org_img : original image (B x 3 x W x H)
        - org_depth : original depth map (B x 1 x W x H)

        (05/14 inhee)
        '''
        views = views.squeeze()
        rotates = views[:,0:3]            # B x 3, (-1.0  ~ 1.0)
        rotates = rotates * 60.0        # B x 3, (-60.0 ~ 60.0)
        #rotates = rotates * 180.0 / math.pi
        trans = views[:,3:6]                # B x 3, (-1.0  ~ 1.0)
        trans = trans / 10.0                # B x 3, (-0.1  ~ 0.1)
        trans[:,2:3]= 0                       # z-translation range (0)


        canon_pc = self.canon_depth_to_3d(canon_depth)
        org_pc = self.canon_3d_to_org_3d(canon_pc, rotates, trans)
        org_depth = self.org_3d_to_org_depth(org_pc)
        warp_grid = self.get_warp_grid(org_depth)
        org_img = self.get_org_image(warp_grid, canon_img)

        return org_img, org_depth