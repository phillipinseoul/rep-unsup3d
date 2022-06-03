'''
Define SIDE, MAD metrics for evaluating the 3D reconstruction accuracy.
(Dataset: BFM synface)
'''
import torch
from .utils import ImageFormation

EPS = 1e-7

class BFM_Metrics():
    def __init__(self, depth_ac, depth_gt, mask, device="cuda"):
        '''
        - depth_ac: depth map of the actual (warped) view
        - depth_gt: ground truth depth map 
        - mask: depth mask from gt_depth and org_depth
        '''
        self.depth_ac = depth_ac          # B x 1 x W x H
        self.depth_gt = depth_gt          # B x 1 x W x H
        self.device = device
        self.mask = mask

    def SIDE_error(self):
        '''SIDE (scale-invariant depth error) between depth maps'''
        _, _, W, H = self.depth_ac.shape

        del_uv = torch.log(self.depth_ac + EPS) - torch.log(self.depth_gt + EPS)
        del_uv = del_uv * self.mask
        
        temp_1 = torch.sum(del_uv ** 2, dim = (1,2,3)) / (W * H)
        temp_2 = torch.sum(del_uv, dim = (1,2,3)) / (W * H)
        
        side = torch.sqrt(temp_1 - (temp_2 ** 2))
        side = (temp_1 - (temp_2 ** 2))

        return side.mean()                         

    def SIDE_error_v2(self):
        del_uv = torch.log(self.depth_ac + EPS) - torch.log(self.depth_gt + EPS)
        del_uv = del_uv * self.mask
        
        temp = torch.sum(del_uv, dim = (1,2,3), keepdim = True) / torch.sum(self.mask, dim = (1,2,3), keepdim=True)
        
        side = torch.sum((del_uv - temp)**2, dim = (1,2,3)) / torch.sum(self.mask, dim = (1,2,3))
        side = torch.sqrt(side)

        return side.mean()

    def MAD_error(self):
        '''MAD (mean angle deviation) between normal maps'''
        imageForm = ImageFormation(device=self.device, size=64)

        # get normal maps
        normal_ac = imageForm.depth_to_normal(self.depth_ac)    # B x 3 x W x H
        normal_gt = imageForm.depth_to_normal(self.depth_gt)    # B x 3 x W x H

        # compute angle deviations
        inner = normal_ac * normal_gt
        angle_rad = torch.arccos(torch.sum(inner, dim=1, keepdim=True))
        angle_deg = torch.rad2deg(angle_rad)
        
        # mean angle deviation
        angle_deg = angle_deg * self.mask
        mad = torch.mean(angle_deg)
        return mad









