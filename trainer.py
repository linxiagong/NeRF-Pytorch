"""NeRF Trainer"""
import os
from collections import defaultdict
from typing import Any

import imageio
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from loss import Loss
from nerf_render import NeRFRender, RayHelper

to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


class Trainer(object):
    def __init__(self,
                 nerf_render: NeRFRender,
                 loss_fn: Loss,
                 optimizer: torch.optim.Optimizer,
                 scheduler,
                 config: dict,
                 H: int,
                 W: int,
                 focal: float,
                 log_dir: str,
                 use_wandb: bool = False,
                 save_best_ckpt_only: bool = True) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nerf_render = nerf_render.to(self.device)
        self.model_name = nerf_render.__class__.__name__
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        # basic train setting
        train_params = config["train_params"]
        self.num_rand = train_params["num_random"]
        self.num_epochs_focus_bbox = train_params["num_epochs_focus_bbox"] or 0
        self.eval_interval = train_params["eval_interval"]
        self.iter_cnt = 0  # current iteration step
        self.start_epoch = 0
        self.eval_loss = np.inf  # current evaluation loss

        self.ndc = config["dataset_params"]["ndc"]
        self._H = int(H)
        self._W = int(W)
        self._focal = focal
        self._K = np.array([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])
        self.white_bkgd = config["dataset_params"]["white_bkgd"]

        # model setting
        self.use_viewdirs = config["model_params"]["use_viewdirs"]

        # logging setting
        self.use_wandb = use_wandb
        self.log_dir = log_dir
        self.save_best_ckpt_only = save_best_ckpt_only
        self.fps = config["dataset_params"]["fps"] or 25

    def sample_rays_for_train(self, epoch: int):
        """Customized policy.
        Focus on different part of the image."""
        H, W = self._H, self._W
        if epoch < self.num_epochs_focus_bbox:
            # temporary setting in hardcode
            dH = int(H // 2 * 0.6)
            dW = int(W // 2 * 0.6)
            coords = torch.stack(
                torch.meshgrid(torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                               torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW)), -1)
        else:
            coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)),
                                 -1)  # (H, W, 2)
        coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
        select_inds = np.random.choice(coords.shape[0], size=[self.num_rand], replace=False)  # (N_rand,)
        select_coords = coords[select_inds].long()  # (N_rand, 2)
        return select_coords

    def train(self, num_epochs: int, dataloader_train: DataLoader, dataloader_val: DataLoader,
              dataloader_test: DataLoader) -> str:
        H, W, focal = self._H, self._W, self._focal
        K = self._K

        eval_loss = self.eval_loss  # np.inf
        for epoch in range(self.start_epoch, num_epochs):
            loss_accumulated = defaultdict(float)
            self.nerf_render.train()

            pbar = tqdm(dataloader_train)
            pbar.set_description(f"Train #{epoch}")
            for i, data in enumerate(pbar):
                image, pose = data["image"], data["pose"]
                rays_o, rays_d = RayHelper.get_rays(H=W, W=W, K=K,
                                                    c2w=torch.Tensor(pose[:3, :4]))  # (H, W, 3), (H, W, 3)
                if self.ndc:
                    rays_o, rays_d = RayHelper.ndc_rays(H=H, W=W, focal=focal, near=1., rays_o=rays_o, rays_d=rays_d)
                # take num_rand rays for training
                select_coords = self.sample_rays_for_train(epoch=epoch)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rgb_gt = torch.Tensor(image)[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

                rgb, disp, acc, all_res = self.nerf_render.render(rays_o=rays_o.to(self.device),
                                                                  rays_d=rays_d.to(self.device),
                                                                  use_viewdirs=self.use_viewdirs,
                                                                  white_bkgd=self.white_bkgd)
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # get loss
                loss, loss_dict = self.loss_fn(pred=all_res, target={"rgb_map": rgb_gt})
                loss.backward()

                loss_accumulated["total"] += loss.item()
                for k in loss_dict.keys():
                    loss_accumulated[k] += loss_dict[k]

                # step optimizer
                self.optimizer.step()
                self.scheduler.step()

                # update logs
                self.iter_cnt += 1
                if (i % 50 == 0 and i > 0) or i == len(dataloader_train) - 1:
                    info = {k: v / (i + 1) for k, v in loss_accumulated.items()}
                    tqdm.write(f"Train #{epoch} i={i}| {info}")
                pbar.set_postfix({f"loss_{k}": v / (i + 1) for k, v in loss_accumulated.items()})

                # evaluate
                if self.iter_cnt % self.eval_interval == 0:
                    new_eval_loss = self.run_eval(dataloader_eval=dataloader_val)
                    # save model if metrics are better
                    if new_eval_loss < eval_loss:
                        self.save_model(epoch=epoch, eval_loss=new_eval_loss)
                        eval_loss = new_eval_loss

        # evaluate when train ends
        new_eval_loss = self.run_eval(dataloader_eval=dataloader_val)
        # save model if metrics are better
        if new_eval_loss < eval_loss:
            self.save_model(epoch=epoch, eval_loss=new_eval_loss)
            eval_loss = new_eval_loss

        return self.ckpt_path

    @torch.no_grad()
    def run_eval(self, dataloader_eval: DataLoader) -> float:
        H, W, focal = self._H, self._W, self._focal
        K = self._K

        rgbs = []
        disps = []
        loss_accumulated = defaultdict(float)
        self.nerf_render.eval()

        pbar = tqdm(dataloader_eval)
        pbar.set_description(f"Validation")
        for i, data in enumerate(pbar):
            image, pose = data["image"], data["pose"]
            # render full image
            rgb, disp, acc, all_res = self.nerf_render.render_full_image(H,
                                                                         W,
                                                                         focal,
                                                                         K,
                                                                         pose,
                                                                         ndc=self.ndc,
                                                                         use_viewdirs=self.use_viewdirs,
                                                                         white_bkgd=self.white_bkgd)
            rgb_gt = torch.Tensor(image).to(self.device)

            loss, loss_dict = self.loss_fn(pred=all_res, target={"rgb_map": rgb_gt})

            # logging
            rgbs.append(rgb.detach().cpu().numpy())
            disps.append(disp.detach().cpu().numpy())
            loss_accumulated["total"] += loss.item()
            for k in loss_dict.keys():
                loss_accumulated[k] += loss_dict[k]
            info = {k: v / (i + 1) for k, v in loss_accumulated.items()}
            tqdm.write(f"Validation| {info}")
            if self.use_wandb:
                for k, v in loss_accumulated.items():
                    wandb.log({f"Validation/loss_{k}": v / (i + 1)}, step=self.iter_cnt)

        rgbs = np.stack(rgbs, 0)
        disps = np.stack(disps, 0)
        imageio.mimwrite(os.path.join(self.log_dir, f'train_{self.iter_cnt}_rgb.mp4'),
                         to8b(rgbs),
                         fps=self.fps,
                         quality=8)
        imageio.mimwrite(os.path.join(self.log_dir, f'train_{self.iter_cnt}_disp.mp4'),
                         to8b(disps / np.max(disps)),
                         fps=self.fps,
                         quality=8)
        print('write video done')

        return loss_accumulated["total"] / (i + 1)

    def save_model(self, epoch: int = None, eval_loss: Any = None):
        """Save model to log_dir."""
        if self.save_best_ckpt_only:
            ckpt_path = os.path.join(self.log_dir, f"{self.model_name}.pth")
        else:
            ckpt_path = os.path.join(self.log_dir, f"{self.model_name}-{self.iter_cnt}.pth")
        torch.save(
            {
                "iter_cnt": self.iter_cnt,
                "epoch": epoch,
                "loss": eval_loss,
                "ndc": self.ndc,
                "use_viewdirs": self.use_viewdirs,
                "white_bkgd": self.white_bkgd,
                "model_state_dict": self.nerf_render.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict()
            },
            ckpt_path,
        )
        self.ckpt_path = ckpt_path
        print(f"Model saved at : {ckpt_path}")

    def load_model(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        self.iter_cnt = int(checkpoint["iter_cnt"])
        self.start_epoch = int(checkpoint["epoch"] or 0)
        self.eval_loss = checkpoint["eval_loss"] or np.inf
        self.nerf_render.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.ckpt_path = ckpt_path
