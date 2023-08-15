"""Just for visualizing."""

import os

import imageio
import numpy as np
import torch
from tqdm import tqdm
from renders import NeRFRender

to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


@torch.no_grad()
def render_video(nerf_render:NeRFRender, H, W, focal, K, poses, ndc: bool, use_viewdirs: bool, white_bkgd: bool, log_dir: str,
                 prefix: str, fps: int):

    rgbs = []
    disps = []
    depths = []
    for pose in tqdm(poses, desc="Render Video"):
        all_res = nerf_render.render_full_image(H,
                                                W,
                                                focal,
                                                K,
                                                pose,
                                                ndc=ndc,
                                                use_viewdirs=use_viewdirs,
                                                white_bkgd=white_bkgd)
        rgb = all_res.get("rgb_map", None)
        if rgb is not None:
            rgbs.append(rgb.detach().cpu().numpy())
        disp = all_res.get("disp_map", None)
        if disp is not None:
            disps.append(disp.detach().cpu().numpy())
        depth = all_res.get("depth_map", None)
        if depth is not None:
            depths.append(depth.detach().cpu().numpy())
    if len(rgbs) > 0: 
        rgbs = np.stack(rgbs, 0)
        rgb_path = os.path.join(log_dir, f'{prefix}_rgb.mp4')
        imageio.mimwrite(rgb_path, to8b(rgbs), fps=fps, quality=8)
    if len(disps) > 0:
        disps = np.stack(disps, 0)
        disp_path = os.path.join(log_dir, f'{prefix}_disp.mp4')
        imageio.mimwrite(disp_path, to8b(disps / np.max(disps)), fps=fps, quality=8)
    if len(depths) > 0:
        depths = np.stack(depths, 0)
        depth_path = os.path.join(log_dir, f'{prefix}_depth.mp4')
        imageio.mimwrite(depth_path, to8b(depths / np.max(depths)), fps=fps, quality=8)
        print(f'Write video done: \n\t{rgb_path}\n\t{disp_path}\n\t{depth_path}')
