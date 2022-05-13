'''
Define the functions for image processing 
in Photo-geometric Autoencoding pipeline
'''

import torch

class ImageFormation():
    def __init__(self, size=64, k_s_max=1.0, k_s_min=0.0, k_d_max=1.0, k_d_min=0.0):
        self.img_size = size
        self.k_s_max = k_s_max
        self.k_s_min = k_s_min
        self.k_d_max = k_d_max
        self.k_d_min = k_d_min
    
    def depth_to_normal(self, depth_map):
        '''
        - input:
            depth_map: B x 1 x W x H
        - output:
            normal_map: B x 3 x W x H
        '''
        B, _,  W, H = depth_map.shape 

        x_range = torch.linspace(0, H - 1, H, dtype = torch.float32)
        y_range = torch.linspace(0, W - 1, W, dtype = torch.float32)

        x_grid, y_grid = torch.meshgrid(x_range, y_range, indexing = 'ij')
        x_grid = x_grid.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)
        y_grid = y_grid.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)

        depth_x_s = torch.zeros_like(depth_map)
        depth_y_s = torch.zeros_like(depth_map)
        depth_x_s[:, :, 1:W, :] = depth_map[:, :, 0:W-1, :]
        depth_y_s[:, :, :, 1:H] = depth_map[:, :, :, 0:H-1]

        v1 = torch.cat([x_grid, (y_grid - 1), depth_y_s], dim = 1)
        v2 = torch.cat([(x_grid - 1), y_grid, depth_x_s], dim = 1)
        c = torch.cat([x_grid, y_grid, depth_map], dim = 1)

        d = torch.cross(v2 - c, v1 - c, dim = 1)
        normal_map = d / torch.sqrt(torch.sum(d ** 2, dim = 1, keepdim=True))

        return normal_map

    def normal_to_shading(self, normal_map, lighting):
        '''
        - input: 
            i) normal_map: B x 3 x W x H
            ii) lighting: B x 4 x 1 x 1
        - output:
            shading_map: B x 1 x W x H
        '''
        B, _, W, H = normal_map.shape

        # light_net outputs: k_s, k_d, l_x, l_y
        k_s, k_d, l_x, l_y = lighting[:, 0:1, :, :], lighting[:, 1:2, :, :], lighting[:, 2:3, :, :], lighting[:, 3:4, :, :]

        # Change range of k_s, k_d: (-1, 1) to (k_s_min, k_s_max)
        k_s = self.k_s_max * (k_s + 1) / 2 + self.k_s_min * (1 - k_s) / 2
        k_d = self.k_d_max * (k_d + 1) / 2 + self.k_d_min * (1 - k_d) / 2

        # Get light direction (l_dir)
        ones = torch.ones(B, 1, 1, 1)
        l_dir = torch.cat((l_x, l_y, ones), 1)
        l_dir /= (l_x ** 2 + l_y ** 2 + 1) ** 0.5   # l_dir: B x 3 x 1 x 1

        # Compute shading value for each pixel
        shading_map = torch.zeros(B, 1, W, H)

        for i in range(1, W - 1):
            for j in range(1, H - 1):
                # Get inner product of light direction and normal
                inner = l_dir * normal_map[:, :, i:i+1, j:j+1]
                inner = torch.sum(inner, dim=1, keepdim=True)
                sh_ij = k_s + k_d * torch.relu(inner)
                shading_map[:, :, i, j] = sh_ij[:, :, 0, 0]

        return shading_map

    def alb_to_canon(self, albedo, shading):
        pass

    

    








 