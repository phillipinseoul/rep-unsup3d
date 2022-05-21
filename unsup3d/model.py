
from locale import normalize
from cv2 import norm
from os.path import join
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.models as pre_model
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from datetime import datetime

from unsup3d.networks import ImageDecomp
from unsup3d.utils import ImageFormation
from unsup3d.renderer import RenderPipeline
from unsup3d.metrics import BFM_Metrics
from unsup3d.utils import get_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PhotoGeoAE(nn.Module):
    def __init__(self, configs):
        super(PhotoGeoAE, self).__init__()

        '''initialize params'''
        self.lambda_p = configs['lambda_p']
        self.lambda_f = configs['lambda_f']
        self.depth_v = configs['depth_v']
        self.alb_v = configs['alb_v']
        self.light_v = configs['light_v']
        self.view_v = configs['view_v']
        self.use_gt_depth = configs['use_gt_depth']
        self.use_conf = configs['use_conf']
        self.b_size = configs['batch_size']
        
        '''
        # Inhee (05/21) I use different path definition here.

        if configs['write_logs']:
            log_dir = join(configs['exp_path'], 'logs', 'exp_' + datetime.now().strftime("%H%M%S"))
            # log_dir = join(configs['exp_path'], 'logs')
            # os.makedirs(log_dir)
            self.logger = SummaryWriter(str(log_dir))
            # self.logger = SummaryWriter(join(configs['exp_path'], 'logs', datetime.now().strftime("%H:%M:%S")))
        
        
        '''


        '''initialize image decomposition networks'''
        self.imgDecomp = ImageDecomp(device=device,
                                     depth_v=self.depth_v, 
                                     alb_v=self.alb_v, 
                                     light_v=self.light_v, 
                                     view_v=self.view_v,
                                     use_conf=self.use_conf)

        self.percep = PercepLoss()

        '''pipeline utils'''
        self.imgForm = ImageFormation(device=device, size=64)
        self.render = RenderPipeline(device=device, b_size=self.b_size)
        

    def get_photo_loss(self, img1, img2, conf):
        L1_loss = torch.abs(img1 - img2)

        losses = torch.log(1/torch.sqrt(2 * torch.pi * conf ** 2)) \
            * torch.exp(-torch.sqrt(torch.Tensor([2]).to(device)) * L1_loss / conf)

        num_cases = img1.shape[1] * img1.shape[2] * img1.shape[3]
        loss = -torch.sum(losses, dim=(1, 2, 3)) / num_cases

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

        '''image decomposition'''
        self.depth = self.imgDecomp.get_depth_map(input)     # B x 3 x W x H
        self.albedo = self.imgDecomp.get_albedo(input)       # B x 1 x W x H 
        self.view = self.imgDecomp.get_view(input)           # B x 6
        self.light = self.imgDecomp.get_light(input)         # B x 4

        raw_conf_percep, raw_conf = self.imgDecomp.get_confidence(input)    # B 2 W/4 H/4 ,, B 2 W H 

        self.conf_percep = raw_conf_percep[:,0:1,:,:]        # B x 1 x W/4 x H/4
        self.conf = raw_conf[:,0:1,:,:]                      # B x 1 x W x H

        self.f_conf_percep = raw_conf_percep[:,1:2,:,:]        # B x 1 x W/4 x H/4
        self.f_conf = raw_conf[:,1:2,:,:]                      # B x 1 x W x H

        self.f_albedo = torch.flip(self.albedo, dims = [3])      # in pytorch, we should flip the last dimension
        self.f_depth = torch.flip(self.depth, dims = [3])        # we made comment as W x H order, but in fact it's H x W (torch default) 
                                                       # So here, I flipped based on last dim

        ############################################ need to check flipping (05/14, inhee) !!!

        '''implement some pipeline'''
        # unflipped case
        self.normal = self.imgForm.depth_to_normal(self.depth)                  # B x 3 x W x H
        self.shading = self.imgForm.normal_to_shading(self.normal, self.light)  # B x 1 x W x H 
        self.canon_img = self.imgForm.alb_to_canon(self.albedo, self.shading)   # B x 3 x W x H

        org_img, org_depth = self.render(canon_depth=self.depth, 
                                         canon_img=self.canon_img, 
                                         views=self.view)

        # flipped case
        self.f_normal = self.imgForm.depth_to_normal(self.f_depth)             # B x 3 x W x H
        self.f_shading = self.imgForm.normal_to_shading(self.f_normal, self.light)  # B x 1 x W x H 
        self.f_canon_img = self.imgForm.alb_to_canon(self.f_albedo, self.f_shading) # B x 3 x W x H

        f_org_img, f_org_depth = self.render(canon_depth=self.f_depth, 
                                             canon_img=self.f_canon_img, 
                                             views=self.view)

        # final results
        self.recon_output = org_img
        self.f_recon_output = f_org_img

        '''calculate loss'''
        self.percep_loss = self.percep(input, self.recon_output, self.conf_percep) # (b_size)
        self.photo_loss = self.get_photo_loss(input, self.recon_output, self.conf)  # (b_size)
        self.org_loss = self.photo_loss + self.lambda_p * self.percep_loss          # (b_size)
        
        self.f_percep_loss = self.percep(input, self.f_recon_output, self.f_conf_percep) # (b_size)
        self.f_photo_loss = self.get_photo_loss(input, self.f_recon_output, self.f_conf)  # (b_size)
        self.flip_loss = self.f_photo_loss + self.lambda_p * self.f_percep_loss           # (b_size)
        
        self.tot_loss = self.org_loss + self.lambda_f * self.flip_loss

        '''for BFM dataset, calculate 3D reconstruction accuracy (SIDE, MAD)'''
        if self.use_gt_depth:
            bfm_metrics = BFM_Metrics(org_depth, gt_depth)
            self.side_error = bfm_metrics.SIDE_error()
            self.mad_error = bfm_metrics.MAD_error()

        '''
        if plot_interms:
            interms = {
                'depth':depth,
                'albedo':albedo,
                'canon_img':canon_img,
                'f_canon_img':f_canon_img,
                'org_depth':org_depth,
                'f_org_depth':f_org_depth,
                'org_img':org_img,
                'f_org_img':f_org_img,
                'input_img':input
            }# intermediate image
            self.visualize(interms)

        losses = {
            'peceploss':perceploss.mean().detach().cpu().item(),
            'photoloss':photoloss.mean().detach().cpu().item(),
            'org_loss':org_loss.mean().detach().cpu().item(),
            'f_peceploss':f_perceploss.mean().detach().cpu().item(),
            'f_photoloss':f_photoloss.mean().detach().cpu().item(),
            'flip_loss':flip_loss.mean().detach().cpu().item(),
            'tot_loss':tot_loss.mean().detach().cpu().item()
        }
        self.logger(losses, step)
        '''


        return self.tot_loss

    def logger(self, losses, step):
        loss_list = list(losses.keys())

        if self.training:
            for loss_name in loss_list:
                self.writer.add_scalar(
                    'Loss_step/train_'+ loss_name,
                    losses[loss_name],
                    step
                )
        else:
            for loss_name in loss_list:
                self.writer.add_scalar(
                    'Loss_step/val_'+ loss_name,
                    losses[loss_name],
                    step
                )
        


    def visualize(self, epoch):
        '''
        all codes for visualization, intermediate outputs
        '''
        def add_image_log(log_path, images, epoch):
            img_grid = torchvision.utils.make_grid(images, normalize=True)
            self.logger.add_image(log_path, img_grid, epoch)

        add_image_log('image_decomposition/depth', self.depth, epoch)
        add_image_log('image_decomposition/albedo', self.albedo, epoch)
        # add_image_log('image_decomposition/view', self.view, epoch)
        # add_image_log('image_decomposition/light', self.light, epoch)

        add_image_log('image_decomposition/conf_percep', self.conf_percep, epoch)
        add_image_log('image_decomposition/conf', self.conf, epoch)
        add_image_log('image_decomposition/f_conf_percep', self.f_conf_percep, epoch)
        add_image_log('image_decomposition/f_conf', self.f_conf, epoch)

        add_image_log('image_decomposition/f_albedo', self.f_albedo, epoch)
        add_image_log('image_decomposition/f_depth', self.f_depth, epoch)

        add_image_log('image_decomposition/normal', self.normal, epoch)
        add_image_log('image_decomposition/shading', self.shading, epoch)
        add_image_log('image_decomposition/canon_img', self.canon_img, epoch)

        add_image_log('image_decomposition/f_normal', self.f_normal, epoch)
        add_image_log('image_decomposition/f_shading', self.f_shading, epoch)
        add_image_log('image_decomposition/f_canon_img', self.f_canon_img, epoch)


    def loss_plot(self, epoch):
        self.logger.add_scalar('losses/percep_loss', torch.mean(self.percep_loss), epoch)
        self.logger.add_scalar('losses/photo_loss', torch.mean(self.photo_loss), epoch)
        self.logger.add_scalar('losses/org_loss', torch.mean(self.org_loss), epoch)

        self.logger.add_scalar('losses/f_percep_loss', torch.mean(self.f_percep_loss), epoch)
        self.logger.add_scalar('losses/f_photo_loss', torch.mean(self.f_photo_loss), epoch)
        self.logger.add_scalar('losses/flip_loss', torch.mean(self.flip_loss), epoch)

        self.logger.add_scalar('losses/tot_loss', torch.mean(self.tot_loss), epoch)

        if self.use_gt_depth:
            self.logger.add_scalar('losses/side_error', self.side_error, epoch)
            self.logger.add_scalar('losses/mad_error', self.mad_error, epoch)


    def set_logger(self, writer):
        self.logger = writer
    def save_results(self):
        pass

class PercepLoss(nn.Module):
    def __init__(self, requires_grad = False):
        super(PercepLoss, self).__init__()
        self.layers = pre_model.vgg16(pretrained=True)
        
        # layer 15's output is ReLU3_3
        modules = [self.layers.features[i] for i in range(16)]
        self.relu3_3 = nn.Sequential(*modules)

        # normalization of input
        self.transforms = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        for module in modules:
            for param in module.parameters():
                param.requires_grad = requires_grad

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
        # print("Feature dim:", n_feat)

        feat_L1 = torch.abs(feat1 - feat2)
        loss = torch.log(1/torch.sqrt(2 * torch.pi * conf ** 2)) \
            * torch.exp(-feat_L1**2/(2*conf**2))
        
        num_cases = feat1.shape[2] * feat1.shape[3] * n_feat
        tot_loss = -torch.sum(loss, dim=(1, 2, 3)) / num_cases

        return tot_loss
        





