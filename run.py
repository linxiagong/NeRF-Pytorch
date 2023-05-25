import os
import shutil
from time import gmtime, strftime

import numpy as np
import torch
import wandb
import yaml
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from loss import NeRFLoss
from nerf_render import NeRFRender
from networks import NeRFFull

to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


def parse_yaml_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Parse YAML configuration file')
    parser.add_argument('--config', type=str, help='Path to the YAML configuration file')
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "render_only"],
    )
    # parser.add_argument("--use_tb", dest="use_tensorboard", default=True action="store_true")
    # parser.add_argument("--use_wandb", dest="use_wandb", default=False, action="store_true")
    args = parser.parse_args()

    config_file = args.config
    config = parse_yaml_config(config_file)

    # ==== Initialize logging ====
    logging_params = config["train_params"]["logging_params"]
    exp_name = config["train_params"]["exp_name"] or os.path.basename(config_file).split(".")[0]
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
    ckpt_path = config["model_params"]["ckpt_path"]  # will load ckpt in trainer

    # ==== Initialize dataset ====
    from datasets import get_dataset
    dataset_train, dataset_val, dataset_test = get_dataset(config["dataset_params"])
    H = int(dataset_train.H)
    W = int(dataset_train.W)
    focal = dataset_train.focal
    dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=1, num_workers=0)
    dataloader_eval = DataLoader(dataset_val, shuffle=True, batch_size=1, num_workers=0)
    dataloader_test = DataLoader(dataset_test, shuffle=False, batch_size=1, num_workers=0)

    # ==== Train ====

    if args.mode == "train":
        from trainer import Trainer

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
                          focal=focal,
                          log_dir=log_dir,
                          save_best_ckpt_only=logging_params["save_best_ckpt_only"])
        trainer.load_model(ckpt_path=ckpt_path)
        ckpt_path = trainer.train(num_epochs=config["train_params"]["num_epochs"],
                                  dataloader_train=dataset_train,
                                  dataloader_val=dataset_val,
                                  dataloader_test=dataset_test)

    # ==== Render with pretrained model ====
    if args.mode == "render_only" or args.mode == "train":
        assert os.path.exists(ckpt_path), f"{ckpt_path} does not exist!"
        print(f'Start rendering with ckpt={ckpt_path}')

        # load checkpoint
        checkpoint = torch.load(ckpt_path)
        nerf_render.load_model(network_params=checkpoint['model_state_dict'])
        nerf_render.eval()
        from render_poses import pose_spherical
        from visualizer import render_video
        render_c2ws = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]],
                                  0)
        render_video(
            nerf_render=nerf_render,
            H=H,
            W=W,
            focal=focal,
            K=np.array([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]]),
            poses=render_c2ws,
            #  ndc=checkpoint["ndc"],
            #  use_viewdirs=checkpoint["use_viewdirs"],
            #  white_bkgd=checkpoint["white_bkgd"],
            ndc=config["dataset_params"]["ndc"],
            use_viewdirs=config["model_params"]["use_viewdirs"],
            white_bkgd=config["dataset_params"]["white_bkgd"],
            log_dir=log_dir,
            prefix=f'render_ckpt_{checkpoint["iter_cnt"]}',
            fps=config["dataset_params"]["fps"])
