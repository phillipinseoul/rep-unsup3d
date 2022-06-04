import torch
import numpy as np
import cv2
import os
from unsup3d.utils import *
import math

############################## borrowed from author's code! ##############################
# https://github.com/elliottwu/unsup3d
class Visualization():
    def __init__(self, model, renderer):
        self.model = model
        self.renderer = renderer

    def translate_pts(self, pts, trans_xyz):
        return pts + trans_xyz

    def rotate_pts(self, pts, rot_mat):
        centroid = torch.FloatTensor([0.,0., self.rot_center_depth]).to(pts.device).view(1,1,3)
        pts = pts - centroid  # move to centroid
        pts = pts.matmul(rot_mat.transpose(2,1))  # rotate
        pts = pts + centroid  # move back
        return pts

    def render_yaw(self, canon_img, depth, v_before=None, v_after=None, rotations=None, maxr=90, nsample=9, crop_mesh=None):
            b, c, h, w = canon_img.shape
            grid_3d = self.renderer.canon_depth_to_3d(depth)

            if crop_mesh is not None:
                top, bottom, left, right = crop_mesh  # pixels from border to be cropped
                if top > 0:
                    grid_3d[:,:top,:,1] = grid_3d[:,top:top+1,:,1].repeat(1,top,1)
                    grid_3d[:,:top,:,2] = grid_3d[:,top:top+1,:,2].repeat(1,top,1)
                if bottom > 0:
                    grid_3d[:,-bottom:,:,1] = grid_3d[:,-bottom-1:-bottom,:,1].repeat(1,bottom,1)
                    grid_3d[:,-bottom:,:,2] = grid_3d[:,-bottom-1:-bottom,:,2].repeat(1,bottom,1)
                if left > 0:
                    grid_3d[:,:,:left,0] = grid_3d[:,:,left:left+1,0].repeat(1,1,left)
                    grid_3d[:,:,:left,2] = grid_3d[:,:,left:left+1,2].repeat(1,1,left)
                if right > 0:
                    grid_3d[:,:,-right:,0] = grid_3d[:,:,-right-1:-right,0].repeat(1,1,right)
                    grid_3d[:,:,-right:,2] = grid_3d[:,:,-right-1:-right,2].repeat(1,1,right)

            grid_3d = grid_3d.reshape(b, -1, 3)
            im_trans = []

            # inverse warp
            if v_before is not None:
                rot_mat, trans_xyz = get_transform_matrices(v_before)
                grid_3d = self.translate_pts(grid_3d, -trans_xyz)
                grid_3d = self.rotate_pts(grid_3d, rot_mat.transpose(2,1))

            if rotations is None:
                rotations = torch.linspace(-math.pi/180*maxr, math.pi/180*maxr, nsample)

            for i, ri in enumerate(rotations):
                ri = torch.FloatTensor([0, ri, 0]).to(canon_img.device).view(1,3)
                rot_mat_i, _ = get_transform_matrices(ri)
                grid_3d_i = self.rotate_pts(grid_3d, rot_mat_i.repeat(b,1,1))

                if v_after is not None:
                    if len(v_after.shape) == 3:
                        v_after_i = v_after[i]
                    else:
                        v_after_i = v_after
                    rot_mat, trans_xyz = get_transform_matrices(v_after_i)
                    grid_3d_i = self.rotate_pts(grid_3d_i, rot_mat)
                    grid_3d_i = self.translate_pts(grid_3d_i, trans_xyz)

                # faces = get_face_idx(b, h, w).to(im.device)
                # textures = get_textures_from_im(im, tx_size=self.tex_cube_size)
                faces = get_faces(b, w, h).to(canon_img.device)
                textures = get_textures_from_im(canon_img, tx_size=2)
                warped_images = self.renderer.renderer.render_rgb(grid_3d_i, faces, textures).clamp(min=-1., max=1.)
                im_trans += [warped_images]

            return torch.stack(im_trans, 1)  # b x t x c x h x w

def get_textures_from_im(canon_img, tx_size=1):
    b, c, h, w = canon_img.shape

    if tx_size == 1:
        textures = torch.cat([canon_img[:,:,:h-1,:w-1].reshape(b,c,-1), canon_img[:,:,1:,1:].reshape(b,c,-1)], 2)
        textures = textures.transpose(2,1).reshape(b,-1,1,1,1,c)
    elif tx_size == 2:
        textures1 = torch.stack([canon_img[:,:,:h-1,:w-1], canon_img[:,:,:h-1,1:], canon_img[:,:,1:,:w-1]], -1).reshape(b,c,-1,3)
        textures2 = torch.stack([canon_img[:,:,1:,:w-1], canon_img[:,:,:h-1,1:], canon_img[:,:,1:,1:]], -1).reshape(b,c,-1,3)
        textures = vcolor_to_texture_cube(torch.cat([textures1, textures2], 2)) # bxnx2x2x2xc
    else:
        raise NotImplementedError("Currently support texture size of 1 or 2 only.")
    return textures

def vcolor_to_texture_cube(vcolors):
    # input bxcxnx3
    b, c, n, f = vcolors.shape
    coeffs = torch.FloatTensor(
        [[ 0.5,  0.5,  0.5],
         [ 0. ,  0. ,  1. ],
         [ 0. ,  1. ,  0. ],
         [-0.5,  0.5,  0.5],
         [ 1. ,  0. ,  0. ],
         [ 0.5, -0.5,  0.5],
         [ 0.5,  0.5, -0.5],
         [ 0. ,  0. ,  0. ]]).to(vcolors.device)
    return coeffs.matmul(vcolors.permute(0,2,3,1)).reshape(b,n,2,2,2,c)

def get_transform_matrices(view):
    b = view.size(0)

    view
    

    rx = view[:,0]
    ry = view[:,1]
    rz = view[:,2]
    trans_xyz = view[:,3:].reshape(b,1,3)
    # rot_mat = get_rotation_matrix(rx, ry, rz)
    rot_mat = get_rot_mat(rx, ry, rz)
    return rot_mat, trans_xyz

############################## borrowed from author's code! ##############################


'''
def save_video(out_fold, frames, fname='image', ext='.mp4', cycle=False):
    os.makedirs(out_fold, exist_ok=True)
    frames = frames.detach().cpu().numpy().transpose(0,2,3,1)  # TxCxHxW -> TxHxWxC
    if cycle:
        frames = np.concatenate([frames, frames[::-1]], 0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'avc1')
    vid = cv2.VideoWriter(os.path.join(out_fold, fname+ext), fourcc, 25, (frames.shape[2], frames.shape[1]))
    [vid.write(np.uint8(f[...,::-1]*255.)) for f in frames]
    vid.release()

def save_image(out_fold, img, fname='image', ext='.png'):
    os.makedirs(out_fold, exist_ok=True)
    img = img.detach().cpu().numpy().transpose(1,2,0)
    if 'depth' in fname:
        im_out = np.uint16(img*65535.)
    else:
        im_out = np.uint8(img*255.)
    cv2.imwrite(os.path.join(out_fold, fname+ext), im_out[:,:,::-1])
'''

'''
    def render_animation(self):
            print(f"Rendering video animations")
            b, _, w, h = self.model.depth.shape

            ## morph from target view to canonical
            morph_frames = 15
            view_zero = torch.FloatTensor([0.15 * np.pi / 180 * 60, 0,0,0,0,0]).to(self.model.depth.device)
            morph_s = torch.linspace(0, 1, morph_frames).to(self.model.depth.device)
            view_morph = morph_s.view(-1, 1 ,1) * view_zero.view(1,1,-1) + (1 - morph_s.view(-1,1,1)) * self.model.view.unsqueeze(0)  # TxBx6

            ## yaw from canonical to both sides
            yaw_frames = 80
            yaw_rotations = np.linspace(-np.pi / 2, np.pi /  2, yaw_frames)
            # yaw_rotations = np.concatenate([yaw_rotations[40:], yaw_rotations[::-1], yaw_rotations[:40]], 0)

            ## whole rotation sequence
            view_after = torch.cat([view_morph, view_zero.repeat(yaw_frames, b, 1)], 0)
            yaw_rotations = np.concatenate([np.zeros(morph_frames), yaw_rotations], 0)

            def rearrange_frames(frames):
                morph_seq = frames[:, :morph_frames]
                yaw_seq = frames[:, morph_frames:]

                out_seq = torch.cat([
                    morph_seq[:,:1].repeat(1,5,1,1,1),
                    morph_seq,
                    morph_seq[:,-1:].repeat(1,5,1,1,1),
                    yaw_seq[:, yaw_frames//2:],
                    yaw_seq.flip(1),
                    yaw_seq[:, :yaw_frames//2],
                    morph_seq[:,-1:].repeat(1,5,1,1,1),
                    morph_seq.flip(1),
                    morph_seq[:,:1].repeat(1,5,1,1,1),
                ], 1)
                return out_seq

            ## textureless shape
            front_light = torch.FloatTensor([0,0,1]).to(self.model.depth.device)
            canon_shape_im = (self.model.normal * front_light.view(1,1,1,3)).sum(3).clamp(min=0).unsqueeze(1)
            canon_shape_im = canon_shape_im.repeat(1,3,1,1) * 0.7
            shape_animation = self.render_yaw(canon_shape_im, self.model.depth, v_after=view_after, rotations=yaw_rotations)  # BxTxCxHxW
            self.shape_animation = rearrange_frames(shape_animation)

            ## normal map
            canon_normal_im = self.model.normal.permute(0,3,1,2) /2+0.5
            normal_animation = self.render_yaw(canon_normal_im, self.model.depth, v_after=view_after, rotations=yaw_rotations)  # BxTxCxHxW
            self.normal_animation = rearrange_frames(normal_animation)

            ## textured
            texture_animation = self.renderer.render_yaw(self.model.canon_im /2+0.5, self.model.depth, v_after=view_after, rotations=yaw_rotations)  # BxTxCxHxW
            self.texture_animation = rearrange_frames(texture_animation)
        '''