from . import grid
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import MLP, Embedder


class ColorNet(nn.Module):
    def __init__(self, color_grid_channels: int, **colornet_params) -> None:
        super().__init__()
        # color emission
        self.color_emission = colornet_params["color_emission"]

        # view directions
        self._use_viewdirs = colornet_params["use_viewdirs"]
        if colornet_params["viewdirs_embedder"]["use_pos_embed"]:
            self.embedder_viewdirs = Embedder(**colornet_params["viewdirs_embedder"])
            viewdirs_dim = self.embedder_viewdirs.out_dim
        else:
            self.embedder_viewdirs = nn.Identity()
            viewdirs_dim = colornet_params["viewdirs_embedder"]["input_dims"]

        # shallow mlp
        if self.color_emission:
            k0_view_dim = color_grid_channels - 3
        else:
            k0_view_dim = color_grid_channels
        self.mlp = MLP(input_dim=k0_view_dim + viewdirs_dim, output_channels=3, **colornet_params)

    def forward(self, k0_view, viewdirs=None):
        if self._use_viewdirs and viewdirs is not None:
            viewdirs_embedded = self.embedder_viewdirs(viewdirs)
            k0_feat = torch.cat([k0_view, viewdirs_embedded], dim=-1)
        else:
            k0_feat = k0_view
        return self.mlp(k0_feat)


class DirectVoxGO(nn.Module):
    """
    Refer to: DirectVoxGO
    """
    def __init__(
        self,
        xyz_min,
        xyz_max,
        num_voxels: int,
        num_voxels_base: int,
        density_grid_params: dict,
        color_grid_params: dict,
        use_densitynet: bool = False,
        densitynet_params: dict = None,
        use_colornet: bool = True,
        colornet_params: dict = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))

        # determine based grid resolution
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1 / 3)

        # determine init grid resolution
        self._set_grid_resolution(num_voxels)

        # --- density related setting ---
        if use_densitynet:
            density_channels = density_grid_params["channels"]
            self.densitynet = MLP(input_dim=density_channels, output_channels=1, **densitynet_params)
        else:
            density_grid_params["channels"] = 1
            self.densitynet = None
        self.density_grid = grid.create_grid(world_size=self.world_size,
                                             xyz_min=self.xyz_min,
                                             xyz_max=self.xyz_max,
                                             **density_grid_params)

        # --- color related setting ---
        if use_colornet:
            # feature voxel grid + shallow MLP  (fine stage)
            color_channels = color_grid_params["channels"]
            self.colornet = ColorNet(color_grid_channels=color_channels, **colornet_params)
        else:
            # color voxel grid  (coarse stage)
            color_grid_params["channels"] = 3
            self.colornet = None
        self.color_grid = grid.create_grid(world_size=self.world_size,
                                           xyz_min=self.xyz_min,
                                           xyz_max=self.xyz_max,
                                           **color_grid_params)

    def _set_grid_resolution(self, num_voxels: int):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1 / 3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        print('dvgo: voxel_size      ', self.voxel_size)
        print('dvgo: world_size      ', self.world_size)
        print('dvgo: voxel_size_base ', self.voxel_size_base)
        print('dvgo: voxel_size_ratio', self.voxel_size_ratio)

    def forward(self, pts: torch.Tensor, viewdirs=None):
        """
        pts: [B * N(not really because it's not uniformly sampled), 3]
        viewdirs: [B, 3]
        """
        # TODO skip known free space, only when densitynet is None

        # query for density
        pts_density = self.density_grid(pts)
        if self.densitynet is not None:
            pts_density = self.densitynet(pts_density)

        # query for color
        k0 = self.color_grid(pts)
        if self.colornet is None:
            # use color grid output as rgb
            pts_color = torch.sigmoid(k0)
        else:
            if self.colornet.color_emission:
                # view-dependent color emission
                k0_view = k0[..., 3:]
                k0_diffuse = k0[..., :3]
                k0_logit = self.colornet(k0_view, viewdirs)
                pts_color = torch.sigmoid(k0_logit + k0_diffuse)
            else:
                k0_view = k0
                k0_logit = self.colornet(k0_view, viewdirs)
                pts_color = torch.sigmoid(k0_logit)

        # return pts_color, pts_density
        return torch.cat([pts_color, pts_density], dim=-1)
