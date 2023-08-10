import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # bins: [B, T], old_z_vals
    # weights: [B, T - 1], bin weights.
    # return: [B, n_samples], new_z_vals

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class RayHelper:
    """Get rays from camera and image."""
    @staticmethod
    def get_rays(H, W, K, c2w):
        c2w = torch.Tensor(c2w[:3, :4])
        i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1,
                                                                          H))  # pytorch's meshgrid has indexing='ij'
        i = i.t()
        j = j.t()
        dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
        # Rotate ray directions from camera frame to the world frame
        rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                           -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = c2w[:3, -1].expand(rays_d.shape)
        return rays_o, rays_d

    @staticmethod
    def ndc_rays(H, W, focal, near, rays_o, rays_d):
        """
        Normalized device coordinates (NDC), used in LLFF dataset.
        """
        # ---------------------------------------
        # More details: https://github.com/bmild/nerf/issues/18
        # ---------------------------------------

        # Shift ray origins to near plane
        t = -(near + rays_o[..., 2]) / rays_d[..., 2]
        rays_o = rays_o + t[..., None] * rays_d

        # Projection
        o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
        o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
        o2 = 1. + 2. * near / rays_o[..., 2]

        d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
        d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
        d2 = -2. * near / rays_o[..., 2]

        rays_o = torch.stack([o0, o1, o2], -1)
        rays_d = torch.stack([d0, d1, d2], -1)

        return rays_o, rays_d

    # @staticmethod
    # def sample_rays(rays_o: torch.Tensor,
    #                 rays_d: torch.Tensor,
    #                 near: torch.Tensor,
    #                 far: torch.Tensor,
    #                 num_samples: int,
    #                 lindisp: bool = False,
    #                 perturb: bool = False) -> torch.Tensor:
    #     """
    #     Compute 3D points along rays.

    #     Args:
    #     - rays_o: Tensor of shape (B, 3) representing the origins of rays.
    #     - rays_d: Tensor of shape (B, 3) representing the directions of rays.
    #     - near: Scalar value representing the near plane of the viewing frustum.
    #     - far: Scalar value representing the far plane of the viewing frustum.
    #     - num_samples: Number of samples to take along each ray for volumetric rendering.
    #     - lindisp: If True, sample linearly in inverse depth rather than in depth.
    #     - perturb: If True, each ray is sampled at stratified random points in time.
        
    #     Returns:
    #     - ray_pts: Tensor of shape (B, num_samples, 3) representing the 3D points along rays.
    #     """
    #     # t_vals = torch.linspace(near, far, num_samples, device=rays_o.device)  # Sample depths along rays
    #     # t_vals = t_vals.expand(rays_o.shape[0], num_samples)  # Repeat for each ray in the batch

    #     # ray_pts = rays_o[:, None, :] + rays_d[:, None, :] * t_vals[:, :, None]

    #     num_rays = rays_o.shape[0]

    #     t_vals = torch.linspace(0., 1., steps=num_samples, device=rays_o.device)
    #     if not lindisp:
    #         z_vals = near * (1. - t_vals) + far * (t_vals)
    #     else:
    #         z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

    #     z_vals = z_vals.expand([num_rays, num_samples])

    #     if perturb:
    #         # get intervals between samples
    #         mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    #         upper = torch.cat([mids, z_vals[..., -1:]], -1)
    #         lower = torch.cat([z_vals[..., :1], mids], -1)
    #         # stratified samples in those intervals
    #         t_rand = torch.rand(z_vals.shape)

    #         z_vals = lower + (upper - lower) * t_rand

    #     ray_pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [B, num_samples, 3]

    #     return ray_pts, z_vals

class NeRFRender(nn.Module):
    """Base NeRF Render. One render per scene"""
    def __init__(self, network: nn.Module, **render_kwargs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = network.to(self.device)

        self._ray_chunck = render_kwargs.get("ray_chunk", 1024 * 32)

        self._near = render_kwargs.get("near", 0.)
        self._far = render_kwargs.get("far", 1.)

        # sampling and upsampling on rays
        self._num_samples = render_kwargs.get("num_samples", 4096)
        self._num_importance = render_kwargs.get("num_importance", 0)
        self._lindisp = render_kwargs.get("lindisp", False)
        self._perturb = render_kwargs.get("perturb ", False)
        self._raw_noise_std = render_kwargs.get("raw_noise_std", 0)

    def load_model(self, ckpt_path=None, network_params=None):
        if ckpt_path is not None and os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            self.network.load_state_dict(checkpoint['model_state_dict'])
        if network_params is not None:
            self.network.load_state_dict(network_params)

    def render_full_image(self,
                          H,
                          W,
                          focal,
                          K,
                          c2w,
                          ndc: bool = False,
                          use_viewdirs: bool = False,
                          white_bkgd: bool = False):
        # !! This may need to be refactored to support other loss types, such as Perceptual Loss
        H, W = int(H), int(W)

        rays_o, rays_d = RayHelper.get_rays(H, W, K, c2w[:3, :4])
        if ndc:
            rays_o, rays_d = RayHelper.ndc_rays(H=H, W=W, focal=focal, near=1., rays_o=rays_o, rays_d=rays_d)
        rgb, disp, acc, all_res = self.render(rays_o=rays_o.to(self.device),
                                              rays_d=rays_d.to(self.device),
                                              use_viewdirs=use_viewdirs,
                                              white_bkgd=white_bkgd)
        return rgb, disp, acc, all_res

    def render(self,
               rays_o,
               rays_d,
               use_viewdirs: bool = False,
               c2w_staticcam=None,
               white_bkgd: bool = False,
               **kwargs):
        """Given origin and direction of rays, render (rgb, disp, ...).
        Args:
        rays: array of shape [2, batch_size, 3]. Ray origin and direction for
            each example in batch.
        c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
        ndc: bool. If True, represent ray origin, direction in NDC coordinates.
        near: float or array of shape [batch_size]. Nearest distance for a ray.
        far: float or array of shape [batch_size]. Farthest distance for a ray.
        use_viewdirs: bool. If True, use viewing direction of a point in space in model.
        c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
        camera while using other c2w argument for viewing directions.
        Returns:
        rgb_map: [batch_size, 3]. Predicted RGB values for rays.
        disp_map: [batch_size]. Disparity map. Inverse of depth.
        acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
        all_ret: dict with everything returned by render_rays().
        """
        # if c2w is not None:
        #     # special case to render full image
        #     rays_o, rays_d = get_rays(H, W, K, c2w)
        # else:
        #     # use provided ray batch
        #     rays_o, rays_d = rays

        if use_viewdirs:
            # provide ray directions as input
            viewdirs = rays_d
            # if c2w_staticcam is not None:
            #     # special case to visualize effect of viewdirs
            #     rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
        else:
            viewdirs = None

        sh = rays_d.shape  # [..., 3]
        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()

        near, far = self._near * torch.ones_like(rays_d[..., :1]), self._far * torch.ones_like(rays_d[..., :1])
        # rays = torch.cat([rays_o, rays_d, near, far], -1)
        # if use_viewdirs:
        # rays = torch.cat([rays, viewdirs], -1)

        # Render and reshape
        all_ret = self.batchify_rays(rays_o=rays_o,
                                     rays_d=rays_d,
                                     near=near,
                                     far=far,
                                     viewdirs=viewdirs,
                                     white_bkgd=white_bkgd,
                                     **kwargs)
        for k in all_ret:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

        # k_extract = ['rgb_map', 'disp_map', 'acc_map']
        rgb_map = all_ret["rgb_map"]
        disp_map = all_ret["disp_map"]
        acc_map = all_ret["acc_map"]
        # ret_list = [all_ret[k] for k in k_extract]
        # extras = {k: all_ret[k] for k in all_ret if k not in k_extract}
        # return ret_list + [extras]
        return rgb_map, disp_map, acc_map, all_ret

    def batchify_rays(self, rays_o, rays_d, near, far, viewdirs, white_bkgd, **kwargs):
        """(batchify_render_rays) Render rays in smaller minibatches to avoid OOM.
        """
        _num_rays = rays_o.shape[0]
        all_result = {}
        for i in range(0, _num_rays, self._ray_chunck):
            # result = self.render_rays(rays_flat[i:i+chunk], **kwargs)
            result = self.render_rays(rays_o=rays_o[i:i + self._ray_chunck],
                                      rays_d=rays_d[i:i + self._ray_chunck],
                                      near=near[i:i + self._ray_chunck],
                                      far=far[i:i + self._ray_chunck],
                                      num_samples=self._num_samples,
                                      lindisp=self._lindisp,
                                      perturb=self._perturb,
                                      num_importance=self._num_importance,
                                      viewdirs=viewdirs[i:i + self._ray_chunck],
                                      raw_noise_std=self._raw_noise_std,
                                      white_bkgd=white_bkgd,
                                      **kwargs)
            for k in result:
                if k not in all_result:
                    all_result[k] = []
                all_result[k].append(result[k])

        all_result = {k: torch.cat(all_result[k], 0) for k in all_result}
        return all_result

    def raw2outputs(self, raw, z_vals, rays_d, raw_noise_std: float = 0, white_bkgd: bool = False):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
            raw_noise_std: std dev of noise added to regularize sigma_a output

        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape).to(dists.device)],
                          -1)  # [N_rays, N_samples]

        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(raw[..., 3].shape) * raw_noise_std

        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha.device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None])

        return rgb_map, disp_map, acc_map, weights, depth_map

    def sample_rays(self,
                    rays_o: torch.Tensor,
                    rays_d: torch.Tensor,
                    near: torch.Tensor,
                    far: torch.Tensor,
                    num_samples: int,
                    lindisp: bool = False,
                    perturb: bool = False) -> torch.Tensor:
        """
        Compute 3D points along rays.

        Args:
        - rays_o: Tensor of shape (B, 3) representing the origins of rays.
        - rays_d: Tensor of shape (B, 3) representing the directions of rays.
        - near: Scalar value representing the near plane of the viewing frustum.
        - far: Scalar value representing the far plane of the viewing frustum.
        - num_samples: Number of samples to take along each ray for volumetric rendering.
        - lindisp: If True, sample linearly in inverse depth rather than in depth.
        - perturb: If True, each ray is sampled at stratified random points in time.
        
        Returns:
        - ray_pts: Tensor of shape (B, num_samples, 3) representing the 3D points along rays.
        """
        # t_vals = torch.linspace(near, far, num_samples, device=rays_o.device)  # Sample depths along rays
        # t_vals = t_vals.expand(rays_o.shape[0], num_samples)  # Repeat for each ray in the batch

        # ray_pts = rays_o[:, None, :] + rays_d[:, None, :] * t_vals[:, :, None]

        num_rays = rays_o.shape[0]

        t_vals = torch.linspace(0., 1., steps=num_samples, device=rays_o.device)
        if not lindisp:
            z_vals = near * (1. - t_vals) + far * (t_vals)
        else:
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

        z_vals = z_vals.expand([num_rays, num_samples])

        if perturb:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape)

            z_vals = lower + (upper - lower) * t_rand

        ray_pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [B, num_samples, 3]

        return ray_pts, z_vals

    def render_rays(self,
                    rays_o,
                    rays_d,
                    near,
                    far,
                    num_samples: int,
                    lindisp: bool = False,
                    perturb: bool = False,
                    num_importance: int = 0,
                    viewdirs=None,
                    raw_noise_std: float = 0,
                    white_bkgd: bool = False):
        """Volume Rendering."""
        # sample points along the given rays
        pts, z_vals = self.sample_rays(rays_o=rays_o,
                                            rays_d=rays_d,
                                            near=near,
                                            far=far,
                                            num_samples=num_samples,
                                            lindisp=lindisp,
                                            perturb=perturb)
        raw = self.network(pts, viewdirs)
        # TODO
        rgb_map, disp_map, acc_map, weights, depth_map = self.raw2outputs(raw, z_vals, rays_d, raw_noise_std,
                                                                          white_bkgd)
        if num_importance > 0:
            # Hierarchical sampling
            rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], num_importance, det=(perturb == 0.))
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[...,
                         None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]

            # will auto switch to coarse if fine_network is not there
            raw = self.network(pts, viewdirs, model_name='fine')

            rgb_map, disp_map, acc_map, weights, depth_map = self.raw2outputs(raw, z_vals, rays_d, raw_noise_std,
                                                                              white_bkgd)

        result = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
        result['raw'] = raw
        if num_importance > 0:
            result['rgb0'] = rgb_map_0
            result['disp0'] = disp_map_0
            result['acc0'] = acc_map_0
            result['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

        for k in result:
            if (torch.isnan(result[k]).any() or torch.isinf(result[k]).any()):
                print(f"! [Numerical Error] {k} contains nan or inf.")
        return result

