"""
DVGO Implementation
Reference: https://github.com/sunset1995/DirectVoxGO
Difference with NeRFRender:
    - DO NOT sample the same amount of points on each ray
        |-> adjusted funcs: sample_rays, raw2alpha, and render_rays
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch.utils.cpp_extension import load
from typing import Tuple

from networks import DirectVoxGO

from .nerf_render import NeRFRender

# -----------------------------------
# load CUDA codes for DVGO rendering
# -----------------------------------
parent_dir = os.path.dirname(os.path.abspath(__file__))
render_utils_cuda = load(name='render_utils_cuda',
                         sources=[
                             os.path.join(parent_dir, path)
                             for path in ['dvgo_cuda/render_utils.cpp', 'dvgo_cuda/render_utils_kernel.cu']
                         ],
                         verbose=True)


class DVGORender(NeRFRender):
    def __init__(self, network: DirectVoxGO, **render_kwargs):
        super().__init__(network, **render_kwargs)

        self._step_size = render_kwargs.get("stepsize", 1)

        # set the alpha values everywhere at the begin of training
        # determine the density bias shift
        alpha_init = render_kwargs.get("alpha_init", 1e-6)
        self.alpha_init = alpha_init
        self.register_buffer('act_shift', torch.FloatTensor([np.log(1 / (1 - alpha_init) - 1)]))
        print('dvgo: set density bias shift to', self.act_shift)

        # the early stage of NeRF optimization is dominated by free space (i.e., space with low density)
        # set fast_color_thres to dilter the scene
        self.fast_color_thres = render_kwargs.get("fast_color_thres", 0)

    def sample_rays(
            self,
            rays_o: torch.Tensor,
            rays_d: torch.Tensor,
            near: float,
            far: float,
            # num_samples: int,
            stepsize: int = 1,
            lindisp: bool = False,
            perturb: bool = False) -> torch.Tensor:
        '''Sample query points on rays.
        All the output points are sorted from near to far.

        Input:
            rays_o, rayd_d:   both in [B, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        near = self._near       # CUDA takes float instead of tensor
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        stepdist = stepsize * self.network.voxel_size.item()

        ray_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max = render_utils_cuda.sample_pts_on_rays(
            rays_o, rays_d, self.network.xyz_min, self.network.xyz_max, near, far, stepdist)
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        ray_id = ray_id[mask_inbbox]
        step_id = step_id[mask_inbbox]
        return ray_pts, ray_id, step_id

    def render_rays(self,
                    rays_o,
                    rays_d,
                    near,
                    far,
                    lindisp: bool = False,
                    perturb: bool = False,
                    viewdirs=None,
                    raw_noise_std: float = 0,
                    white_bkgd: bool = False):
        """Volume Rendering of Voxel-NeRF."""
        # sample points along the given rays
        ray_pts, ray_id, step_id = self.sample_rays(
            rays_o=rays_o,
            rays_d=rays_d,
            near=near,
            far=far,
            stepsize=self._step_size,
        )

        # TODO skip known free space
        # if self.mask_cache is not None:
        #     mask = self.mask_cache(ray_pts)
        #     ray_pts = ray_pts[mask]
        #     ray_id = ray_id[mask]
        #     step_id = step_id[mask]

        # predict density and color
        raw = self.network(ray_pts, viewdirs[ray_id])  # (rgb(3), density(1))

        rgb_map, disp_map, acc_map, weights, depth_map = self.raw2outputs(raw, ray_id, step_id, N=len(rays_o))

        result = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map, 'depth_map': depth_map}
        return result

    def raw2outputs(
            self,
            raw: Tuple[torch.Tensor, torch.Tensor],  # raw=(rgb, density/sigma)
            ray_id,
            step_id,
            N: int,  # N=len(rays_o)
    ):
        """
        final color c = sum (T_i * alpha_i * c_i), 1<=i<=n (pts on the ray)
                    where T_i = product (1 - alpha_j), 1<=j<=i-1
        """
        # rgb, sigma = torch.split(raw, [3, 1], dim=-1)
        rgb, sigma = raw
        sigma = sigma.squeeze()  # for latter operations

        # query for alpha w/ post-activation
        interval = self._step_size * self.network.voxel_size_ratio
        alpha = self.activate_density(sigma, interval)
        if self.fast_color_thres > 0:
            mask = (alpha > self.fast_color_thres)
            rgb = rgb[mask]
            # sigma = sigma[mask]
            alpha = alpha[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]


        # compute accumulated transmittance
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask = (weights > self.fast_color_thres)
            rgb = rgb[mask]
            # sigma = sigma[mask]
            alpha = alpha[mask]
            weights = weights[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            

        # Ray marching
        rgb_marched = torch_scatter.segment_coo(src=(weights.unsqueeze(-1) * rgb),
                                                index=ray_id,
                                                out=torch.zeros([N, 3], device=self.device),
                                                reduce='sum')
        depth = torch_scatter.segment_coo(src=(weights * step_id), index=ray_id, out=torch.zeros([N], device=self.device), reduce='sum')
        # TODO add color of background
        # rgb_marched += (alphainv_last.unsqueeze(-1) * render_kwargs['bg'])

        disp_map = None
        acc_map = None
        return rgb_marched, disp_map, acc_map, weights, depth

    def activate_density(self, density:torch.Tensor, interval=None):
        interval = interval if interval is not None else self.network.voxel_size_ratio
        shape = density.shape
        return Raw2Alpha.apply(density.flatten().contiguous(), self.act_shift, interval).reshape(shape)

    def density(self, pts):
        return self.network.density(pts)

class Raw2Alpha(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density, shift, interval):
        '''
        alpha = 1 - exp(-softplus(density + shift) * interval)
              = 1 - exp(-log(1 + exp(density + shift)) * interval)
              = 1 - exp(log(1 + exp(density + shift)) ^ (-interval))
              = 1 - (1 + exp(density + shift)) ^ (-interval)
        '''
        exp, alpha = render_utils_cuda.raw2alpha(density, shift, interval)
        if density.requires_grad:
            ctx.save_for_backward(exp)
            ctx.interval = interval
        return alpha

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        '''
        alpha' = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)'
               = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)
        '''
        exp = ctx.saved_tensors[0]
        interval = ctx.interval
        return render_utils_cuda.raw2alpha_backward(exp, grad_back.contiguous(), interval), None, None


class Alphas2Weights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, ray_id, N):
        weights, T, alphainv_last, i_start, i_end = render_utils_cuda.alpha2weight(alpha, ray_id, N)
        if alpha.requires_grad:
            ctx.save_for_backward(alpha, weights, T, alphainv_last, i_start, i_end)
            ctx.n_rays = N
        return weights, alphainv_last

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_weights, grad_last):
        alpha, weights, T, alphainv_last, i_start, i_end = ctx.saved_tensors
        grad = render_utils_cuda.alpha2weight_backward(alpha, weights, T, alphainv_last, i_start, i_end, ctx.n_rays,
                                                       grad_weights, grad_last)
        return grad, None, None
