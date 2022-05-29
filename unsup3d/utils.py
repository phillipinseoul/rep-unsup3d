'''
Define the functions for image processing 
in Photo-geometric Autoencoding pipeline
'''

import torch
import torch.nn.functional as F
import math

EPS = 1e-7

class ImageFormation():
    def __init__(self, device, size=64, 
                    k_s_max=1.0, k_s_min=0.0, k_d_max=1.0, k_d_min=0.0):
        W = 64
        H = 64
        theta_fov = 10

        c_u = (W-1)/2
        c_v = (H-1)/2
        f = (W-1)/(2*math.tan(deg2rad(theta_fov)/2))

        self.K = torch.tensor([
            [f, 0, c_u],
            [0, f, c_v],
            [0, 0, 1]], dtype = torch.float32)              
        self.K_inv = torch.linalg.inv(self.K)               # 3x3 matrix
        self.K = self.K.unsqueeze(0).to(device)

        self.img_size = size
        self.k_s_max = k_s_max
        self.k_s_min = k_s_min
        self.k_d_max = k_d_max
        self.k_d_min = k_d_min
        self.device = device
    
    
    def depth_to_normal(self, depth_map):
        '''
        - input:
            depth_map: B x 1 x W x H
        - output:
            normal_map: B x 3 x W x H
        '''
        
        '''changed depth_to_normal by including K_inv (05/24, Yuseung)'''
        grid = gen_grid(depth_map)
        grid_3d = torch.cat(
            [grid, torch.ones_like(depth_map, dtype=torch.float32).to(self.device)],
            dim=1)
        depth_pc = safe_matmul(grid_3d, self.K_inv.transpose(2, 1)) * depth_map

        v1 = depth_pc[:, :, 1:-1, 2:] - depth_pc[:, :, 1:-1, :-2]
        v2 = depth_pc[:, :, 2:, 1:-1] - depth_pc[:, :, :-2, 1:-1]
        normal = v1.cross(v2, dim=1)
        # normal = torch.cross(v2, -v1, dim=1)

        zero_pad = nn.ZeroPad2d(1)
        normal = zero_pad(normal)
        normal = normal / (torch.sqrt(torch.sum(normal ** 2, dim=1, keepdim=True)) + EPS)

        return normal

    def normal_to_shading(self, normal_map, lighting):
        '''
        - input: 
            i) normal_map: B x 3 x W x H
            ii) lighting: B x 4
        - output:
            shading_map: B x 1 x W x H
        '''
        B, _, W, H = normal_map.shape

        # light_net outputs: k_s, k_d, l_x, l_y
        k_s, k_d, l_x, l_y = lighting[:, 0:1], lighting[:, 1:2], lighting[:, 2:3], lighting[:, 3:4]

        # Change range of k_s, k_d: (-1, 1) to (k_s_min, k_s_max)
        k_s = self.k_s_max * (k_s + 1) / 2 + self.k_s_min * (1 - k_s) / 2
        k_d = self.k_d_max * (k_d + 1) / 2 + self.k_d_min * (1 - k_d) / 2

        # Get light direction (l_dir)
        ones = torch.ones(B, 1, device=self.device)
        l_dir = torch.cat((l_x, l_y, ones), dim=1) / torch.sqrt(l_x ** 2 + l_y ** 2 + 1)     # l_dir: B x 3
        l_dir = l_dir.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, W, H)


        shading_map = F.relu(torch.sum(l_dir * normal_map, dim=1, keepdim=True)) * k_d.view(B,1,1,1) + k_s.view(B,1,1,1)

        return shading_map


    def alb_to_canon(self, albedo, shading_map):
        '''
        - input:
            i) albedo: B x 3 x W x H
            ii) shading_map: B x 1 x W x H
        - output:
            canon_view: B x 3 x W x H
        '''
        # calculate the canonical view
        canon_view = albedo * shading_map

        return canon_view * 2. -1.


def get_mask(depth):
    '''
    - depth: B x 1 x W x H
    '''
    ones = torch.ones_like(depth, dtype=torch.float32)
    mask = depth > torch.min(depth)
    mask_init = mask * ones

    '''erode the mask'''
    mask_avg = torch.nn.functional.avg_pool2d(
        mask_init, kernel_size=3, stride=1, padding=1
    )
    mask_avg = mask_avg * mask_init
    mask_erode = (mask_avg > 0.8) * ones

    return mask_erode
 

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

    xx,yy = torch.meshgrid(x_coord, y_coord, indexing = 'ij')
    xx = xx.unsqueeze(0).unsqueeze(0).repeat(B,1,1,1) # B x 1 x W x H
    yy = yy.unsqueeze(0).unsqueeze(0).repeat(B,1,1,1) # B x 1 x W x H

    return torch.cat([yy,xx], dim = 1).to(img.device)  # B x 2 x W x H


def safe_matmul(first, second):
    '''
    input:
    - first: first matrix to be multiplied
    - second: second matrix to be multiplied
    output:
    - res: matrix multiplied results. (B x 3 x W x H)

    one of matrix should be "3x3" and other should be "B x 3 x W x H"
    Other input shapes would be asserted (05/13 inhee)
    '''
    # case handling. 
    if len(first.shape) < 4 and len(second.shape) == 4:
        first_is_3x3 = True

        if second.shape[1] != 3:
            print("improper shape")
            print("first shape: ", first.shape)
            print("second shape: ", second.shape)

    elif len(first.shape)==4 and len(second.shape) < 4:
        first_is_3x3 = False
        
        if first.shape[1] != 3:
            print("improper shape")
            print("first shape: ", first.shape)
            print("second shape: ", second.shape)

    else:
        print("improper shape")
        print("first shape: ", first.shape)
        print("second shape: ", second.shape)
        assert(0)

    # device different case
    if first.device != second.device:
        print("different device")
        assert(0)
    

    if first_is_3x3:
        second_temp = second.reshape(second.shape[0], second.shape[1],-1)   # B x 3 x WH
        
        res = torch.matmul(first, second_temp)  # B x 3 x WH (matmul broadcasting)
        res = res.reshape(second.shape[0], second.shape[1], second.shape[2], second.shape[3])

    else:
        first_temp = first.reshape(first.shape[0], first.shape[1], -1)  # B x 3 x WH
        first_temp = first_temp.permute(0,2,1)                          # B x WH x 3
        res = torch.matmul(first_temp, second)  # B x WH x 3
        res = res.permute(0,2,1)                # B x 3 x WH
        res = res.reshape(first.shape[0], first.shape[1], first.shape[2], first.shape[3])

    return res


def get_rot_mat(alpha, beta, gamma):
    '''
    input:
    - alpha : rotation in xy plane (deg) (B x 1)
    - beta : rotation in yz plane (deg) (B x 1)
    - gamma : rotation in zx plane (deg) (B x 1)
    ouptut:
    - rot_mat: rotation matrix (B x 3 x 3)


    ################################################# YOU NEED TO TEST IT WHETHER IT'S PROPER (05/13) #######
    '''

    # generate rotation matrix
    device = alpha.device
    a = alpha/180 * torch.pi
    b = beta/180 * torch.pi
    g = gamma/180 * torch.pi
    
    ## <rot_a>
    ##
    ## cos(alpha) -sin(alpha)   0
    ## sin(alpha)  cos(alpha)   0
    ##     0           0        1
    ##
    s_a = torch.sin(a)  # B x 1
    c_a = torch.cos(a)  # B x 1
    rot_a = torch.cat(
        [
            torch.stack([c_a, -s_a, torch.zeros_like(c_a, dtype=torch.float32).to(device)], dim = 2), # B x 1 x 3
            torch.stack([s_a, c_a, torch.zeros_like(c_a, dtype=torch.float32).to(device)], dim = 2),  # B x 1 x 3
            torch.stack([
                torch.zeros_like(c_a, dtype = torch.float32),
                torch.zeros_like(c_a, dtype = torch.float32),
                torch.ones_like(c_a, dtype = torch.float32)
            ], dim = 2).to(device)                                                      # B x 1 x 3  
        ], dim = 1
    )                                                                                                 # B x 3 x 3
    ## <rot_b>
    ##
    ##     1         0          0
    ##     0     cos(beta) -sin(beta)
    ##     0     sin(beta)  cos(beta)
    ##
    s_b = torch.sin(b)  # B x 1
    c_b = torch.cos(b)  # B x 1
    rot_b = torch.cat(
        [
            torch.stack([
                torch.ones_like(c_b, dtype = torch.float32),
                torch.zeros_like(c_b, dtype = torch.float32),
                torch.zeros_like(c_b, dtype = torch.float32)
            ], dim = 2).to(device),                                                     # B x 1 x 3 
            torch.stack([torch.zeros_like(c_b, dtype=torch.float32).to(device), c_b, -s_b], dim = 2), # B x 1 x 3
            torch.stack([torch.zeros_like(c_b, dtype=torch.float32).to(device), s_b, c_b], dim = 2)  # B x 1 x 3
        ], dim = 1
    )  
    ## <rot_g>
    ##
    ##  cos(gamma)    0      sin(gamma)
    ##      0         1          0
    ## -sin(gamma)    0      cos(gamma)
    ##
    s_g = torch.sin(g)  # B x 1
    c_g = torch.cos(g)  # B x 1
    rot_g = torch.cat(
        [
            torch.stack([c_g, torch.zeros_like(c_g, dtype=torch.float32).to(device), s_g], dim = 2),  # B x 1 x 3
            torch.stack([
                torch.zeros_like(c_g, dtype = torch.float32),
                torch.ones_like(c_g, dtype = torch.float32),
                torch.zeros_like(c_g, dtype = torch.float32)
            ], dim = 2).to(device),                                                     # B x 1 x 3 
            torch.stack([-s_g, torch.zeros_like(c_g, dtype=torch.float32).to(device), c_g], dim = 2)  # B x 1 x 3
        ], dim = 1
    )  
    rot_mat = torch.matmul(rot_a, torch.matmul(rot_g,rot_b))

    return rot_mat


def get_faces(B, W, H):
    '''
    input:
    - B, W, H: size of input shape
    output:
    - faces : faces (B x 2(W-1)(H-1) x 3)

    it returns faces
    
    divide faces like this (following tessellation in paper)
    ####
    ####  (i-1, j) *  *  *  *   (i,j)           
    ####      *   *      R        *
    ####      *       *           *
    ####      *   L       *       *
    ####  (i-1, j-1) *  *  *  * (i,j-1)     
    ####
    
    faces are consists of coordinate(order) of 3 vertices
    order is counter clock-wise (right-thumb's rule)
    (05/14 inhee)
    '''

    faces = torch.arange(W*H, dtype = torch.int32)
    faces_2D = faces.reshape(W,H)   # W x H

    face_L = torch.stack(
        [
            faces_2D[0:W-1, 1:H],   # (W-1) x (H-1)
            faces_2D[0:W-1, 0:H-1], # (W-1) x (H-1)
            faces_2D[1:W, 0:H-1]    # (W-1) x (H-1)
        ],
        dim = -1
    )   # (W-1) x (H-1) x 3
    face_R = torch.stack(
        [
            faces_2D[0:W-1, 1:H],   # (W-1) x (H-1)
            faces_2D[1:W, 0:H-1],   # (W-1) x (H-1)
            faces_2D[1:W, 1:H]      # (W-1) x (H-1)
        ],
        dim = -1
    )   # (W-1) x (H-1) x 3

    faces = torch.cat(
        [
            face_L.reshape(-1, 3),
            face_R.reshape(-1, 3)
        ],
        dim = 0
    )   # 2(W-1)(H-1) x 3

    return faces.unsqueeze(0).repeat(B, 1, 1)      # B x 2(W-1)(H-1) x 3





 