import numpy as np
import torch
from tqdm import tqdm

from utils import functimer


class RayHelper:
    @staticmethod
    def get_rays(H,
                 W,
                 K,
                 c2w,
                 flip_x: bool = False,
                 flip_y: bool = False,
                 inverse_y: bool = False,
                 mode: str = 'center'):
        """
        Get rays from camera and image.
        Input:
            - H, W
            - K
            - c2w
            - flip_x, flip_y: bool. Set True to support co3d
            - inverse_y: bool. Set True to support blendedmvs, nsvf, tankstemple
        """
        c2w = torch.Tensor(c2w[:3, :4])
        i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1,
                                                                          H))  # pytorch's meshgrid has indexing='ij'
        i = i.t().float()
        j = j.t().float()

        # deal with different modes
        if mode == 'lefttop':
            pass
        elif mode == 'center':
            i, j = i + 0.5, j + 0.5
        elif mode == 'random':
            i = i + torch.rand_like(i)
            j = j + torch.rand_like(j)
        else:
            raise NotImplementedError

        # if inverse_y:
        #     dirs = torch.stack([(i - K[0][2]) / K[0][0], (j - K[1][2]) / K[1][1], torch.ones_like(i)], -1)
        # else:
        #     dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
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


def _compute_bbox_by_cam_frustrm_bounded(dataset_params: dict, H: int, W: int, focal: float,
                                         dataloader_train: torch.utils.data.DataLoader, near: float, far: float):
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min

    pbar = tqdm(dataloader_train)
    pbar.set_description(f"Computes the (bounded) bounding box")
    for i, data in enumerate(pbar):
        pose = data["pose"][0]
        # [ATTENTION] we deal with only data with same (H, W, K)
        K = np.array([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])
        rays_o, rays_d = RayHelper.get_rays(H=W, W=W, K=K, c2w=torch.Tensor(pose[:3, :4]))  # (H, W, 3), (H, W, 3)
        if dataset_params.get("ndc", False):
            rays_o, rays_d = RayHelper.ndc_rays(H=H, W=W, focal=focal, near=1., rays_o=rays_o, rays_d=rays_d)
            pts_nf = torch.stack([rays_o + rays_d * near, rays_o + rays_d * far])
        else:
            viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
            pts_nf = torch.stack([rays_o + viewdirs * near, rays_o + viewdirs * far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0, 1, 2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0, 1, 2)))
    return xyz_min, xyz_max


def _compute_bbox_by_cam_frustrm_unbounded(dataset_params: dict, H: int, W: int, focal: float,
                                           dataloader_train: torch.utils.data.DataLoader, near_clip):
    # Find a tightest cube that cover all camera centers
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min

    pbar = tqdm(dataloader_train)
    pbar.set_description(f"Computes the (unbounded) bounding box")
    for i, data in enumerate(pbar):
        pose = data["pose"]

        K = np.array([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])
        rays_o, rays_d = RayHelper.get_rays(H=W, W=W, K=K, c2w=torch.Tensor(pose[:3, :4]))  # (H, W, 3), (H, W, 3)
        if dataset_params.get("ndc", False):
            rays_o, rays_d = RayHelper.ndc_rays(H=H, W=W, focal=focal, near=1., rays_o=rays_o, rays_d=rays_d)

        pts = rays_o + rays_d * near_clip
        xyz_min = torch.minimum(xyz_min, pts.amin((0, 1)))
        xyz_max = torch.maximum(xyz_max, pts.amax((0, 1)))
    center = (xyz_min + xyz_max) * 0.5
    radius = (center - xyz_min).max() * dataset_params.get("unbounded_inner_r", 1.0)
    xyz_min = center - radius
    xyz_max = center + radius
    return xyz_min, xyz_max

@functimer
def compute_bbox_by_cam_frustrm(dataset_params: dict, H: int, W: int, focal: float,
                                dataloader_train: torch.utils.data.DataLoader, near: float, far: float,
                                **kwargs):
    if dataset_params.get("unbounded_inward", False):
        xyz_min, xyz_max = _compute_bbox_by_cam_frustrm_unbounded(
            dataset_params=dataset_params, 
            H=H, W=W, focal=focal,
            dataloader_train=dataloader_train,
            near_clip=kwargs.get('near_clip', None))
    else:
        xyz_min, xyz_max = _compute_bbox_by_cam_frustrm_bounded(
            dataset_params=dataset_params,
            H=H, W=W, focal=focal,
            dataloader_train=dataloader_train,
            near=near, far=far
        )
    print(f'compute_bbox_by_cam_frustrm: xyz_min={xyz_min}, xyz_max={xyz_max}')
    return xyz_min, xyz_max



from renders import BaseRender

@functimer
@torch.no_grad()
def compute_bbox_by_pretrained_model(model:BaseRender, bbox_thres:float):
    """(Only applicable to Voxel based NeRF)
    """
    world_size = model.network.world_size
    interp = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, world_size[0]),
        torch.linspace(0, 1, world_size[1]),
        torch.linspace(0, 1, world_size[2]),
    ), -1)
    dense_xyz = model.network.xyz_min * (1-interp) + model.network.xyz_max * interp
    density = model.density(dense_xyz)
    alpha = model.activate_density(density)
    mask = (alpha > bbox_thres)
    active_xyz = dense_xyz[mask]
    xyz_min = active_xyz.amin(0)
    xyz_max = active_xyz.amax(0)
    print(f'compute_bbox_by_pretrained_model: xyz_min={xyz_min}, xyz_max={xyz_max}')
    return xyz_min, xyz_max