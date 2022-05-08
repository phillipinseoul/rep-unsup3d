'''
Define the functions for image processing 
in Photo-geometric Autoencoding pipeline
'''

class ImageFormation():
    def __init__(self):
        pass
    
    def depth_to_normal(self, depth_map):
        '''
        - input: depth_map (size: torch.Size([1, 64, 64]))
        - output: normal_map (size: torch.Size([1, 64, 64])
        '''
        d, w, h = depth_map.shape
        normal_map = depth_map.repeat(3, 1, 1)

        # calculate normal map from depth map
        for i in range(1, w - 1):
            for j in range(1, h - 1):
                v1 = torch.Tensor([i, j - 1, depth_map[0][i][j - 1]])
                v2 = torch.Tensor([i - 1, j, depth_map[0][i - 1][j]])
                c = torch.Tensor([i, j, depth[0][i][j]])

                d = torch.cross(v2 - c, v1 - c)
                n = d / torch.sqrt(torch.sum(d ** 2))
                normal_map[:, i, j] = n

        return normal_map

    def normal_to_shading(self, normal_map, lighting):
        pass

    def alb_to_canon(self, albedo, shading):
        pass

    

    








 