
from locale import normalize
from cv2 import norm
from os.path import join
import os
import math
import torch
import torch.nn as nn
import torchvision
import torchvision.models as pre_model
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from datetime import datetime
from pytictoc import TicToc

from unsup3d.networks import ImageDecomp
from unsup3d.utils import ImageFormation
from unsup3d.renderer import *
from unsup3d.metrics import BFM_Metrics
from unsup3d.utils import get_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-7
author_test = False


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

        '''initialize image decomposition networks'''
        self.imgDecomp = ImageDecomp(
            device=device,
            W = 64, H =64,
            depth_v=self.depth_v, 
            alb_v=self.alb_v, 
            light_v=self.light_v, 
            view_v=self.view_v,
            use_conf=self.use_conf
            )

        self.percep = PercepLoss()

        '''pipeline utils'''
        self.imgForm = ImageFormation(device=device, size=64)
        self.render = RenderPipeline(device=device, b_size=self.b_size)
        

    def get_photo_loss(self, img1, img2, conf, mask = None):
        L1_loss = torch.abs(img1 - img2)
        
        losses = L1_loss *2**0.5 / (conf + EPS) + torch.log(conf + EPS)

        if mask is not None:
            losses = losses * mask
            loss = torch.sum(losses, dim = (1,2,3)) / (torch.sum(mask, dim = (1,2,3))+EPS)
        else:
            num_cases = img1.shape[1] * img1.shape[2] * img1.shape[3]
            loss = torch.sum(losses, dim=(1, 2, 3)) / num_cases
        

        return loss



    def forward(self, input):
        '''
        input:
        - input: (Bx3xHxW), preprocessed on dataloader as H=W=64
        implement pipeline here
        '''

        
        '''for BFM datasets, separate gt_depth (05/23 yuseung)'''

        if self.use_gt_depth:
            input, self.gt_depth = input

        '''normalize the input (-1, 1)'''
        input = input * 2.-1.

        '''image decomposition'''
        self.input = input
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
        if author_test:
            self.normal = self.render.depth2normal_author(self.depth)
        else:
            self.normal = self.imgForm.depth_to_normal(self.depth)                  # B x 3 x W x H
        self.shading = self.imgForm.normal_to_shading(self.normal, self.light)  # B x 1 x W x H 
        self.canon_img = self.imgForm.alb_to_canon(self.albedo, self.shading)   # B x 3 x W x H

        org_img, org_depth = self.render(canon_depth=self.depth, 
                                         canon_img=self.canon_img, 
                                         views=self.view)

        # flipped case
        if author_test:
            self.f_normal = self.render.depth2normal_author(self.f_depth)
        else:
            self.f_normal = self.imgForm.depth_to_normal(self.f_depth)             # B x 3 x W x H
        self.f_shading = self.imgForm.normal_to_shading(self.f_normal, self.light)  # B x 1 x W x H 
        self.f_canon_img = self.imgForm.alb_to_canon(self.f_albedo, self.f_shading) # B x 3 x W x H

        f_org_img, f_org_depth = self.render(canon_depth=self.f_depth, 
                                             canon_img=self.f_canon_img, 
                                             views=self.view)

        # final results
        self.recon_output = org_img
        self.f_recon_output = f_org_img

        self.org_depth = org_depth
        self.f_org_depth = f_org_depth

        # Mask of loss
        if USE_WIDER_DEPTH: 
            depth_margin = 0.2
        else:
            depth_margin = 0.1

        mask_org_depth = (org_depth < (1.1 + depth_margin)).float()
        mask_flip_depth = (f_org_depth < (1.1 + depth_margin)).float()
        mask_depth = mask_org_depth * mask_flip_depth       #b 1 w h
        mask_depth = mask_depth.detach()
        self.mask_depth = mask_depth

        self.recon_output = self.recon_output * self.mask_depth
        self.f_recon_output = self.f_recon_output * self.mask_depth         # (added 05/29 inhee) it would affect the percep loss only.

        '''calculate loss'''
        self.L1_loss = torch.abs(self.recon_output - input).mean()
        self.effective_L1_loss = (torch.abs(self.recon_output-input)*self.mask_depth).sum()/self.mask_depth.sum()
        self.effective_pixels = self.mask_depth.sum()

        self.percep_loss = self.percep(input, self.recon_output, self.conf_percep, mask_depth) # (b_size)
        self.photo_loss = self.get_photo_loss(input, self.recon_output, self.conf, mask_depth)  # (b_size)
        self.org_loss = self.photo_loss + self.lambda_p * self.percep_loss          # (b_size)
        
        self.f_percep_loss = self.percep(input, self.f_recon_output, self.f_conf_percep, mask_depth) # (b_size)
        self.f_photo_loss = self.get_photo_loss(input, self.f_recon_output, self.f_conf, mask_depth)  # (b_size)
        self.flip_loss = self.f_photo_loss + self.lambda_p * self.f_percep_loss           # (b_size)
        
        self.tot_loss = self.org_loss + self.lambda_f * self.flip_loss

        '''for BFM dataset, calculate 3D reconstruction accuracy (SIDE, MAD)'''
        if self.use_gt_depth:
            bfm_metrics = BFM_Metrics(org_depth, self.gt_depth)
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
        
        if self.tot_loss.isnan().sum() != 0:
            assert(0)
        elif self.tot_loss.isinf().sum() != 0:
            assert(0)
        

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
        def add_image_log(log_path, images, epoch, normalize=True):
            img_grid = torchvision.utils.make_grid(images, normalize=normalize)
            self.logger.add_image(log_path, img_grid, epoch)

        add_image_log('image_decomposition/depth', (self.depth -0.9)*5.0, epoch, False)
        add_image_log('image_decomposition/albedo', self.albedo, epoch, False)
        # add_image_log('image_decomposition/view', self.view, epoch)
        # add_image_log('image_decomposition/light', self.light, epoch)

        add_image_log('image_decomposition/conf_percep', self.conf_percep, epoch)
        add_image_log('image_decomposition/conf', self.conf, epoch)
        add_image_log('image_decomposition/f_conf_percep', self.f_conf_percep, epoch)
        add_image_log('image_decomposition/f_conf', self.f_conf, epoch)

        add_image_log('image_decomposition/f_albedo', self.f_albedo, epoch, False)
        add_image_log('image_decomposition/f_depth', (self.f_depth-0.9)*5.0, epoch, False)

        add_image_log('image_decomposition/normal', self.normal, epoch)
        add_image_log('image_decomposition/shading', self.shading/2., epoch, False)
        add_image_log('image_decomposition/canon_img', self.canon_img, epoch, False)

        add_image_log('image_decomposition/f_normal', self.f_normal, epoch)
        add_image_log('image_decomposition/f_shading', self.f_shading/2., epoch, False)
        add_image_log('image_decomposition/f_canon_img', self.f_canon_img, epoch, False)

        add_image_log('to_debug/recon_img', (self.recon_output+1.)/2., epoch, False)
        add_image_log('to_debug/f_recon_img', (self.f_recon_output+1.)/2., epoch, False)
        add_image_log('to_debug/input_img', (self.input+1.)/2., epoch, False)
        add_image_log('to_debug/depth_mask', self.mask_depth, epoch)

        add_image_log('image_decomposition/input_img', self.input, epoch)

        '''
        if self.use_gt_depth:
            add_image_log('image_decomposition/gt_depth', self.gt_depth, epoch)
        '''

        add_image_log('reconstruction/recon_output', self.recon_output, epoch)
        add_image_log('reconstruction/f_recon_output', self.f_recon_output, epoch)
        add_image_log('to_debug/org_depth', (self.org_depth - 0.8)*2.5, epoch, False)
        add_image_log('to_debug/f_org_depth', (self.f_org_depth - 0.8)*2.5, epoch, False)

        print('views angles: \n', (self.view.detach().cpu()[0:5, 0:3] * 60.))
        print('views trans: \n', (self.view.detach().cpu()[0:5, 3:] * 0.1))
        print('lights: \n', (self.light.detach().cpu()[0:5,0:2] + 1.)/2.)
        print('normal value range:', self.normal.min().item(), self.normal.max().item())
        print('shading value range:', self.shading.min().item(), self.shading.max().item())
        print('canon_img value range:', self.canon_img.min().item(), self.canon_img.max().item())
        print('recon img value range: ', self.recon_output.min().item(), self.recon_output.max().item())


    def loss_plot(self, epoch):
        self.logger.add_scalar('losses/percep_loss', torch.mean(self.percep_loss*self.lambda_p), epoch)
        self.logger.add_scalar('losses/photo_loss', torch.mean(self.photo_loss), epoch)
        self.logger.add_scalar('losses/org_loss', torch.mean(self.org_loss), epoch)

        self.logger.add_scalar('losses/f_percep_loss', torch.mean(self.f_percep_loss*self.lambda_p*self.lambda_f), epoch)
        self.logger.add_scalar('losses/f_photo_loss', torch.mean(self.f_photo_loss*self.lambda_f), epoch)
        self.logger.add_scalar('losses/flip_loss', torch.mean(self.flip_loss*self.lambda_f), epoch)

        self.logger.add_scalar('losses/tot_loss', torch.mean(self.tot_loss), epoch)
        self.logger.add_scalar('losses/L1_loss', self.L1_loss, epoch)

        self.logger.add_scalar('debug/effective_L1', self.effective_L1_loss, epoch )
        self.logger.add_scalar('debug/effective_pixels', self.effective_pixels, epoch)
        self.logger.add_scalar('debug/ambient_light_mean(bg)', ((self.light.detach().cpu()[:,0:1] + 1.)/2.).mean(), epoch)
        self.logger.add_scalar('debug/ambient_light_std(bg)', ((self.light.detach().cpu()[:,0:1] + 1.)/2.).std(), epoch)

        self.logger.add_scalar('debug/diffusion_light_mean', ((self.light.detach().cpu()[:,1:2] + 1.)/2.).mean(), epoch)
        self.logger.add_scalar('debug/diffusion_light_std', ((self.light.detach().cpu()[:,1:2] + 1.)/2.).std(), epoch)

        self.logger.add_scalar('debug/view angles (abs mean)', (self.view.detach().cpu()[:, 0:3] * 60.).abs().mean(), epoch)
        self.logger.add_scalar('debug/view ranss (abs mean)', (self.view.detach().cpu()[:, 3:5] * 0.1).abs().mean(), epoch)
        
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

    def forward(self, img1, img2, conf, mask = None):
        '''
        input:
        - img1: input image (Bx3xHxW)
        - img2: output image (Bx3XHxW)
        - conf: confidence map (Bx1x(H/4)x(W/4))
        - mask: mask of loss (B 1 H W)
        output:
        - loss: perceptual loss (real number)
        calculate PercepLoss based on L1 distance & confidence
        '''
        img1 = (img1+1.)/2.
        img2 = (img2+1.)/2.
        
        n_img1 = self.transforms(img1)
        n_img2 = self.transforms(img2)

        feat1 = self.relu3_3(n_img1)
        feat2 = self.relu3_3(n_img2)

        n_feat = feat1.shape[1]

        feat_Loss = (feat1 - feat2) ** 2
        loss = feat_Loss / (2*conf**2 + EPS) + torch.log(conf + EPS)

        if mask is not None:
            bm, _, hm, wm = mask.shape
            b, _, h, w = loss.shape
            
            re_mask = nn.functional.avg_pool2d(mask, kernel_size = (hm//h, wm//w), stride = (hm//h, wm//w)).expand_as(loss)
            tot_loss = torch.sum(loss*re_mask, dim = (1,2,3)) / (torch.sum(re_mask, dim = (1,2,3))+EPS)

        else:
            num_cases = feat1.shape[2] * feat1.shape[3] * n_feat
            tot_loss = torch.sum(loss, dim=(1, 2, 3)) / num_cases

        return tot_loss
        





