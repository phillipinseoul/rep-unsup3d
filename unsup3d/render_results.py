import torch
import numpy as np
import cv2
import os
from unsup3d.utils import *
from unsup3d.renderer import RenderPipeline
import math

class Visualization():
    def __init__(self, model, renderer):
        self.model = model
        self.views = model.view
        self.renderer = renderer

    def render_result(self, canon_img, depth):
            b, c, h, w = canon_img.shape

            views = self.views.squeeze()
            rotates = views[:,0:3]            # B x 3, (-1.0  ~ 1.0)
            rotates = rotates * 60.0          # B x 3, (-60.0 ~ 60.0)

            canon_pc = self.renderer.canon_depth_to_3d(depth)
            img_trans = []

            rot_init = torch.FloatTensor([60., 0., 0.]).repeat(b, 1).to(views.device)

            for i in range(1, 5):
                rot = rot_init / i

                org_pc = self.renderer.canon_3d_to_org_3d(canon_pc, rot, trans)
                org_depth = self.renderer.org_3d_to_org_depth(org_pc)
                warp_grid = self.renderer.get_warp_grid(org_depth)
                org_img = self.renderer.get_org_image(warp_grid, canon_img)
                img_trans.append(org_img)

            return torch.stack(img_trans, 1)        # b x t x c x h x w