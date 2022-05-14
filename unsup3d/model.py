
from cv2 import norm
import torch
import torch.nn as nn
import torchvision.models as pre_model
import torchvision.transforms as transforms
import tensorboardX

from unsup3d.networks import ImageDecomp, ConfNet_v1
from unsup3d.utils import ImageFormation
from unsup3d.renderer import RenderPipeline
from unsup3d.metrics import BFM_Metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PhotoGeoAE():
    def __init__(self, configs):
        '''initialize params'''
        # self.lambda_p = 0.5
        # self.lambda_f = 1.0

        '''TODO: set configs'''

        self.lambda_p = configs['lambda_p']
        self.lambda_f = configs['lambda_f']
        self.depth_v = configs['depth_v']
        self.alb_v = configs['light_v']
        self.view_v = configs['view_v']
        self.use_gt_depth = configs['use_gt_depth']
        
        '''TODO: set configs'''

        '''initialize image decomposition networks'''
        self.imgDecomp = ImageDecomp(self.depth_v, self.alb_v, self.light_v, self.view_v)

        '''TODO: implement ImageDecomp with additional network versions.'''

        self.netC = ConfNet_v1()
        self.percep = PercepLoss()

        '''pipeline utils'''
        self.imgForm = ImageFormation(size=64)
        self.render = RenderPipeline(device=device)
        

    def get_photo_loss(self, img1, img2, conf):
        L1_loss = torch.abs(img1-img2)
        losses = torch.log(torch.sqrt(2*torch.pi*conf**2)) \
            * torch.exp(-torch.sqrt(2)*L1_loss/conf)

        num_cases = img1.shape[1]*img1.shape[2]*img1.shape[3]
        loss = -torch.sum(losses, dim=(1,2,3)) / num_cases

        return loss


    def forward(self, input):
        '''
        input:
        - input: (Bx3xHxW), preprocessed on dataloader as H=W=64
        implement pipeline here
        '''

        '''for BFM datasets, separate gt_depth'''
        if self.use_gt_depth:
            input, gt_depth = input

        albedo = self.imgDecomp.get_depth_map(input)    # B x 3 x W x H
        depth = self.imgDecomp.get_albedo(input)        # B x 1 x W x H 
        view = self.imgDecomp.get_view(input)           # B x 6 x 1 x 1
        light = self.imgDecomp.get_light(input)         # B x 4 x 1 x 1

        raw_conf_percep, raw_conf = self.netC(input)    # B 2 W/4 H/4 ,, B 2 W H 

        conf_percep = raw_conf_percep[:,0:1,:,:]        # B x 1 x W/4 x H/4
        conf = raw_conf[:,0:1,:,:]                      # B x 1 x W x H

        f_conf_percep = raw_conf_percep[:,1:2,:,:]        # B x 1 x W/4 x H/4
        f_conf = raw_conf[:,1:2,:,:]                      # B x 1 x W x H

        f_albedo = torch.flip(albedo, dims = [3])      # in pytorch, we should flip the last dimension
        f_depth = torch.flip(depth, dims = [3])        # we made comment as W x H order, but in fact it's H x W (torch default) 
                                                       # So here, I flipped based on last dim

        ############################################ need to check flipping (05/14, inhee) !!!

        '''implement some pipeline'''
        # unflipped case
        normal = self.imgForm.depth_to_normal(depth)             # B x 3 x W x H
        shading = self.imgForm.normal_to_shading(normal, light)  # B x 1 x W x H 
        canon_img = self.imgForm.alb_to_canon(albedo, shading)   # B x 3 x W x H
        org_img, org_depth = self.render(depth, canon_img, view)

        # flipped case
        f_normal = self.imgForm.depth_to_normal(f_depth)             # B x 3 x W x H
        f_shading = self.imgForm.normal_to_shading(f_normal, light)  # B x 1 x W x H 
        f_canon_img = self.imgForm.alb_to_canon(f_albedo, f_shading) # B x 3 x W x H

        f_org_img, f_org_depth = self.render(f_depth, f_canon_img, view)

        # final results
        recon_output = org_img
        f_recon_output = f_org_img

        '''calculate loss'''
        percep_loss = self.percep(input, recon_output, conf_percep) # (b_size)
        photoloss = self.get_photo_loss(input, recon_output, conf)  # (b_size)
        org_loss = photoloss + self.lambda_p * percep_loss          # (b_size)
        
        f_percep_loss = self.percep(input, f_recon_output, f_conf_percep) # (b_size)
        f_photoloss = self.get_photo_loss(input, f_recon_output, f_conf)  # (b_size)
        flip_loss = f_photoloss + self.lambda_p * f_percep_loss           # (b_size)
        
        tot_loss = org_loss + self.lambda_f * flip_loss

        '''for BFM dataset, calculate 3D reconstruction accuracy (SIDE, MAD)'''
        if use_gt_depth:
            bfm_metrics = BFM_Metrics(org_depth, gt_depth)
            self.side_error = bfm_metrics.SIDE_error()
            self.mad_error = bfm_metrics.MAD_error()

        return tot_loss


    def visualize(self):
        '''
        all codes for visualization, intermediate outputs
        '''
        pass

    def save_results(self):
        pass



class PercepLoss(nn.Module):
    def __init__(self, requires_grad = False):
        super(PercepLoss,self).__init__()
        self.layers = pre_model.vgg16(pretrained=True)
        
        # layer 15's output is ReLU3_3
        modules = [self.layers.features[i] for i in range(16)]
        self.relu3_3 = nn.Sequential(*modules)

        # normalization of input
        self.transforms = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


    def forward(self, img1, img2, conf):
        '''
        input:
        - img1: input image (Bx3xHxW)
        - img2: output image (Bx3XHxW)
        - conf: confidence map (Bx1x(H/4)x(W/4))
        output:
        - loss: perceptual loss (real number)
        calculate PercepLoss based on L1 distance & confidence
        '''
        n_img1 = self.transforms(img1)
        n_img2 = self.transforms(img2)

        feat1 = self.relu3_3(n_img1)
        feat2 = self.relu3_3(n_img2)

        n_feat = feat1.shape[1]
        print("Feature dim:", n_feat)

        feat_L1 = torch.abs(feat1-feat2)
        loss = torch.log(torch.sqrt(2*torch.pi*conf**2)) \
            * torch.exp(-feat_L1**2/(2*conf**2))
        
        num_cases = feat1.shape[2]*feat1.shape[3]*n_feat
        tot_loss = -torch.sum(loss, dim=(1,2,3)) / num_cases

        return tot_loss
        





