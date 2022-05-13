import torch
import math
from neural_renderer import Renderer


class RenderPipeline():
    def __init__(self, device = torch.device("cpu"), args = None):
        '''
        this part should be received from argument "args"
        We need to update it! (05/13 inhee)
        '''
        W = 64
        H = 64
        theta_fov = 10  # 10 degree, not rad
        self.device = device
        self.min_depth = 0.9    # from Table.6 of paper
        self.max_depth = 1.1    # from Table.6 of paper


        # define inverse function
        c_u = (W-1)/2
        c_v = (H-1)/2
        f = (W-1)/(2*math.tan(deg2rad(theta_fov)/2))

        self.K = torch.Tensor([[f, 0, c_u],
                  [0, f, c_v],
                  [0, 0, 1]], dtype = torch.float32).to(self.device)

        self.K_inv = torch.linalg.inv(self.K)               # 3x3 matrix


        # define neural renderer
        # Here, we give camer information by K, I omitted camera relevant args (05/13)
        T = torch.zeros(1,3,dtype = torch.float32).to(self.device)  # zero translation for renderer
        R = torch.eye(3,dtype = torch.float32).to(self.device)      # zero rotation for renderer

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
            dist_coeffs=None,                   # vector of distortion coefficients, default : None 
            orig_size=W,                        # original size of image captured by the camera 
            near = self.min_depth,              # minmum depth
            far = self.max_depth,               # maximum depth
            light_intensity_ambient=1.0, 
            light_intensity_directional=0.,
            light_color_ambient=[1,1,1], 
            light_color_directional=[1,1,1],
            light_direction=[0,1,0]
        )


        


    def canon_depth_to_3d(self, depth):
        '''
        input:
        - depth : depth map (B x 1 x W x H)
        output:
        - can_pc : canoncial point cloud (B x 3 x W x H)

        it changes 1D depth map to 3D point cloud
        output = depth * inv_K * (u,v,1)
        (05/13 inhee)
        '''
        
        pass



    def canon_3d_to_org_3d(self, canon_pc):
        '''
        input:
        - canon_pc : canonical point cloud (B x 3 x W x H)
        output:
        - org_pc : original point cloud (B x 3 x W x H)

        it applies camera viewpoint, by rotating & translating
        output = Rot ( input ) + Trans

        (05/13 inhee)
        '''

        pass


    def org_3d_to_org_depth(self, org_pc):
        '''
        input:
        - org_pc : original point cloud (B x 3 x W x H)
        output:
        - org_depth : original depth map (B x 1 x W x H)

        it render pointcloud with Renderer.
        (05/13 inhee)
        '''

        pass


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

        pass



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

        pass






def deg2rad(deg):
    return deg*math.pi/180

def gen_grid(img):
    '''
    input:
    - img: image or depth map, (B x () x W x H)
    output:
    - grid: tensor holding coordinate of pixel (B x 2 x W x H)

    it makes tensor in same device of input img.
    '''

    if len(img.shape) != 4:
        print("can't generate grid, it's dimension isn't 4")
        print("input shape", img.shape)
        assert(0)
    
    B, _ , W, H = img.shape

    x_coord = torch.linspace(0, W-1, W, dtype = torch.float32)
    y_coord = torch.linspace(0, H-1, H, dtype = torch.float32)





