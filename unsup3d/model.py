
import torch
import torch.nn as nn
import torchvision.models as pre_model
import torchvision.transforms as transforms
import tensorboardX

from unsup3d.networks import ImageDecomp
from unsup3d.utils import ImageFormation

class PhotoGeoAE():
    def __init__(self, depth_v='v0', alb_v='v0', light_v='v0', view_v='v0'):
        '''initialize params'''
        self.lambda_p = 0.5
        self.lambda_f = 1.0

        '''initialize image decomposition networks'''
        self.imgDecomp = ImageDecomp(depth_v, alb_v, light_v, view_v)

        '''TODO: implement ImageDecomp with additional network versions.'''

        self.netC = ConfNet_v1()
        self.percep = PercepLoss()

        '''pipeline utils'''
        self.imgForm = ImageFormation(size=64)

        

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

        albedo = self.imgDecomp.get_depth_map(input)    # B x 3 x W x H
        depth = self.imgDecomp.get_albedo(input)        # B x 1 x W x H 
        view = self.imgDecomp.get_view(input)           # B x 6 x 1 x 1
        light = self.imgDecomp.get_light(input)         # B x 4 x 1 x 1

        conf_percep, conf = self.netC(input)# b 1 H/4 W/4 .. b 1 H W 

        '''implement some pipeline'''

        recon_output = None
        flipped_recon_ouptut = None

        '''calculate loss here'''
        percep_loss = self.percep(input, recon_output, conf_percep) # (b_size)
        photoloss = self.get_photo_loss(input, recon_output, conf)  # (b_size)
        org_loss = photoloss + self.lambda_p * percep_loss          # (b_size)
        
        f_percep_loss = self.percep(input, recon_output, conf_percep) # (b_size)
        f_photoloss = self.get_photo_loss(input, recon_output, conf)  # (b_size)
        flip_loss = f_photoloss + self.lambda_p * f_percep_loss       # (b_size)
        
        tot_loss = org_loss + self.lambda_f * flip_loss

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
        





