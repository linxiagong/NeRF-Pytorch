"""Rewrite load_blender.py"""
import json
import os
import imageio

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

trans_t = lambda t: torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([[1, 0, 0, 0], [0, np.cos(phi), -np.sin(phi), 0], [0, np.sin(
    phi), np.cos(phi), 0], [0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor([[np.cos(th), 0, -np.sin(th), 0], [0, 1, 0, 0], [np.sin(
    th), 0, np.cos(th), 0], [0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


class BlenderDataset(Dataset):
    def __init__(self, root_dir: str, split: str, white_bkgd: bool = False, skip: int = 1, **kwargs):
        assert split in ['train', 'val', 'test']
        self._split = split
        self._root_dir = root_dir
        self._skip = 1 if split == 'train' else skip

        with open(os.path.join(root_dir, f'transforms_{split}.json'), 'r') as fp:
            meta = json.load(fp)

        self._frames = meta['frames'][::self._skip]

        fname = os.path.join(self._root_dir, self._frames[0]['file_path'] + '.png')
        _ = imageio.imread(fname)
        self.H, self.W = _.shape[:2]
        self.camera_angle_x = float(meta['camera_angle_x'])
        self.focal = .5 * self.W / np.tan(.5 * self.camera_angle_x)

        self.near = 2.
        self.far = 6.

    # def __repr__(self):
    #     pass

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, index):

        frame = self._frames[index]
        # image
        fname = os.path.join(self._root_dir, frame['file_path'] + '.png')
        img = imageio.imread(fname)
        img = np.array(img / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
        # pose
        pose = np.array(frame['transform_matrix'], dtype=np.float32)

        return {'image': img, 'pose': pose}

    def dataloader(self) -> DataLoader:
        loader = DataLoader(self, shuffle=(self._split == "train"), batch_size=1, num_workers=0)
        return loader
