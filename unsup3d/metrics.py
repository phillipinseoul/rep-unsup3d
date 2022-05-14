'''
Define SIDE, MAD metrics for evaluating the 3D reconstruction accuracy.
(Dataset: BFM synface)
'''
import torch
from utils import ImageFormation

class BFM_Metrics():
    def __init__(self, depth_ac, depth_gt):
        '''
        - depth_ac: depth map of the actual (warped) view
        - depth_gt: ground truth depth map 
        '''
        self.depth_ac = depth_ac          # B x 1 x W x H
        self.depth_gt = depth_gt          # B x 1 x W x H

    def SIDE_error(self):
        '''SIDE (scale-invariant depth error) between depth maps'''
        _, _, W, H = self.depth_ac.shape

        del_uv = torch.log(self.depth_ac) - torch.log(self.depth_gt)
        temp_1 = torch.sum(del_uv ** 2) / (W * H)
        temp_2 = torch.sum(del_uv) / (W * H)
        side = torch.sqrt(temp_1 - (temp_2 ** 2))
        return side                         

    def MAD_error(self):
        '''MAD (mean angle deviation) between normal maps'''
        imageForm = ImageFormation()

        # get normal maps
        normal_ac = imageForm.depth_to_normal(self.depth_ac)    # B x 3 x W x H
        normal_gt = imageForm.depth_to_normal(self.depth_gt)    # B x 3 x W x H

        # compute angle deviations
        inner = normal_ac * normal_gt
        angle_rad = torch.arccos(torch.sum(inner, dim=1, keepdim=True))
        angle_deg = torch.rad2deg(angle_rad)
        
        # mean angle deviation
        mad = torch.mean(angle_deg)
        return mad









