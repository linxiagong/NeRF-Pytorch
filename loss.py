""" Build loss function.
Ref: https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/run_nerf.py#L814-L824
"""

from collections import defaultdict
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

img2mse = lambda x, y: torch.mean((x - y)**2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))


class Loss(nn.Module):
    def __init__(self, loss_weights: dict) -> None:
        super().__init__()
        self.loss_weights = defaultdict(float)
        self.loss_weights.update(loss_weights)

    def __repr__(self):
        fmt_str = f"[{self.__class__.__name__}] Loss weights:\n"
        for k, v in self.loss_weights.items():
            fmt_str += f"\t{k}: {v}\n"
        return fmt_str

    def forward(self, pred, target):
        raise NotImplementedError()


class NeRFLoss(Loss):
    def __init__(self, loss_weights: dict) -> None:
        super().__init__(loss_weights)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, pred: dict, target: dict) -> Tuple[torch.Tensor, dict]:
        loss = {}

        # MSE
        loss["mse"] = img2mse(pred["rgb_map"][..., :3].to(self.device), target["rgb_map"][..., :3].to(self.device))
        loss["psnr"] = mse2psnr(loss["mse"].detach().cpu())  # does not add into loss

        # MSE0
        if "rgb0" in pred:
            loss["mse0"] = img2mse(pred["rgb0"][..., :3].to(self.device), target["rgb_map"][..., :3].to(self.device))
            loss["psnr0"] = mse2psnr(loss["mse0"].detach().cpu())  # does not add into loss

        total_loss = sum([loss[k] * self.loss_weights[k] for k in loss.keys() if k in self.loss_weights])
        loss = {key: np.squeeze(value.detach().cpu().numpy()) for key, value in loss.items()}
        return total_loss, loss
