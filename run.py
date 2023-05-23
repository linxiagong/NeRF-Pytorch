import os
import shutil
from collections import defaultdict
from time import gmtime, strftime

import numpy as np
import torch
import tqdm
import yaml
from torch.optim.lr_scheduler import ExponentialLR

from loss import Loss, NeRFLoss
from nerf_render import NeRFRender, RayHelper
from networks import NeRFFull


class Trainer(object):
    def __init__(self, nerf_render: NeRFRender, loss_fn: Loss, optimizer: torch.optim.Optimizer, scheduler,
                 config: dict, H: int, W: int, focal: float) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nerf_render = nerf_render.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        # basic train setting
        train_params = config["train_params"]
        self.num_rand = train_params["num_random"]
        self.num_epochs_focus_bbox = train_params["num_epochs_focus_bbox"] or 0

        self.ndc = config["dataset_params"]["ndc"]
        self._H = int(H)
        self._W = int(W)
        self._focal = focal
        self._K = np.array([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])
        self.white_bkgd = config["dataset_params"]["white_bkgd"]

        # model setting
        self.use_viewdirs = config["model_params"]["use_viewdirs"]

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

    def train_one_epoch(self, epoch: int, dataset_train):
        H, W, focal = self._H, self._W, self._focal
        K = self._K

        loss_accumulated = defaultdict(float)
        self.nerf_render.train()

        pbar = tqdm.tqdm(dataset_train)
        pbar.set_description(f"Train #{epoch}")
        for i, data in enumerate(pbar):
            image, pose = data["image"], data["pose"]
            rays_o, rays_d = RayHelper.get_rays(H=W, W=W, K=K, c2w=torch.Tensor(pose[:3, :4]))  # (H, W, 3), (H, W, 3)
            if self.ndc:
                rays_o, rays_d = RayHelper.ndc_rays(H=H, W=W, focal=focal, near=1., rays_o=rays_o, rays_d=rays_d)
            # take num_rand rays for training
            select_coords = self.sample_rays_for_train(epoch=epoch)
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            rbg_gt = torch.Tensor(image)[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

            rgb, disp, acc, all_res = self.nerf_render.render(rays_o=rays_o.to(self.device),
                                                              rays_d=rays_d.to(self.device),
                                                              use_viewdirs=self.use_viewdirs,
                                                              white_bkgd=self.white_bkgd)
            # zero the parameter gradients
            self.optimizer.zero_grad()

            # get loss
            loss, loss_dict = self.loss_fn(pred=all_res, target={"rgb_map": rbg_gt})
            loss.backward()

            loss_accumulated["total"] += loss.item()
            for k in loss_dict.keys():
                loss_accumulated[k] += loss_dict[k]

            # step optimizer
            self.optimizer.step()
            self.scheduler.step()

            # update logs
            if (i % 50 == 0 and i > 0) or i == len(dataset_train) - 1:
                info = {k: v / (i + 1) for k, v in loss_accumulated.items()}
                tqdm.tqdm.write(f"Train #{epoch} i={i}| {info}")
            pbar.set_postfix({f"loss_{k}": v / (i + 1) for k, v in loss_accumulated.items()})
        return loss_dict

    def train(self, num_epochs: int, dataset_train, dataset_val, dataset_test):
        eval_loss = np.inf
        for epoch in range(num_epochs):
            loss_dict = self.train_one_epoch(epoch=epoch, dataset_train=dataset_train)
            # TODO logging

            # # evaluate
            # new_eval_loss = self.run_eval()
            # # save model if metrics are better
            # if new_eval_loss < eval_loss:
            #     self.save_model()
            #     eval_loss = new_eval_loss

    @torch.no_grad()
    def run_eval(self, dataset_eval):
        pass

    def save_model(self):
        pass


if __name__ == '__main__':

    def parse_yaml_config(file_path):
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    import argparse
    parser = argparse.ArgumentParser(description='Parse YAML configuration file')
    parser.add_argument('--config', type=str, help='Path to the YAML configuration file')
    parser.add_argument(
        "--mode",
        default="train",
        choices=[
            "train",
        ],
    )
    # parser.add_argument("--use_wandb", dest="use_wandb", action="store_true")
    args = parser.parse_args()

    config_file = args.config
    config = parse_yaml_config(config_file)

    # ==== Initialize logging ====
    logging_params = config["train_params"]["logging_params"]
    exp_name = os.path.basename(config_file).split(".")[0]
    # logtime_str = strftime("%y_%m_%d_%H.%M.%S", gmtime())
    # exp_name += '-' + logtime_str

    log_dir = os.path.join(logging_params["log_dir"], exp_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f'[Logging Config] log_dir={log_dir}, save_best_ckpt_only:{logging_params["save_best_ckpt_only"]}')
    # dump config into log dir
    shutil.copy(config_file, log_dir)

    # ==== Initialize model ====
    network = NeRFFull(config["model_params"])
    nerf_render = NeRFRender(network=network, **config["render_params"])
    # load weights

    # ==== Initialize dataset ====
    from datasets import get_dataset
    dataset_train, dataset_val, dataset_test = get_dataset(config["dataset_params"])
    H = int(dataset_train.H)
    W = int(dataset_train.W)
    focal = dataset_train.focal

    # ==== Train ====

    # -- create optimizer --
    lr = config["train_params"]["lr"]
    optimizer = torch.optim.Adam(params=nerf_render.parameters(), lr=lr, betas=(0.9, 0.999))
    ###   update learning rate   ###
    decay_rate = 0.1
    # decay_steps = args.lrate_decay * 1000
    # new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = new_lrate
    decay_steps = config["train_params"]["lr_decay"] * 1000
    scheduler = ExponentialLR(optimizer, gamma=decay_rate**(1 / decay_steps))

    # -- get loss --
    loss_fn = NeRFLoss(loss_weights=config["train_params"]["loss_weights"])
    trainer = Trainer(nerf_render=nerf_render,
                      loss_fn=loss_fn,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      config=config,
                      H=H,
                      W=W,
                      focal=focal)
    if args.mode == "train":
        trainer.train(num_epochs=config["train_params"]["num_epochs"],
                      dataset_train=dataset_train,
                      dataset_val=dataset_val,
                      dataset_test=dataset_test)
