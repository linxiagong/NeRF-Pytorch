"""Just for visualizing."""

import os

import imageio
import numpy as np
import torch
from tqdm import tqdm

to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


@torch.no_grad()
def render_video(nerf_render, H, W, focal, K, poses, ndc: bool, use_viewdirs: bool, white_bkgd: bool, log_dir: str,
                 prefix: str, fps: int):

    rgbs = []
    disps = []
    for pose in tqdm(poses, desc="Render Video"):
        rgb, disp, acc, all_res = nerf_render.render_full_image(H,
                                                                W,
                                                                focal,
                                                                K,
                                                                pose,
                                                                ndc=ndc,
                                                                use_viewdirs=use_viewdirs,
                                                                white_bkgd=white_bkgd)
        rgbs.append(rgb.detach().cpu().numpy())
        disps.append(disp.detach().cpu().numpy())
    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    rgb_path = os.path.join(log_dir, f'{prefix}_rgb.mp4')
    imageio.mimwrite(rgb_path, to8b(rgbs), fps=fps, quality=8)
    disp_path = os.path.join(log_dir, f'{prefix}_disp.mp4')
    imageio.mimwrite(disp_path, to8b(disps / np.max(disps)), fps=fps, quality=8)
    print(f'Write video done: \n\t{rgb_path}\n\t{disp_path}')
