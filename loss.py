""" Build loss function """

from collections import defaultdict
from typing import Tuple

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

    def forward(self, pred: dict, target: dict) -> Tuple[torch.Tensor, dict]:
        loss = {}

        # MSE
        loss["mse"] = img2mse(pred["rgb_map"], target["rgb_map"])

        # MSE0
        if "rbg0" in pred:
            loss["mse0"] = img2mse(pred["rgb0"], target["rgb_map"])

        total_loss = sum([loss[k] * self.loss_weights[k] for k in loss.keys()])
        loss = {key: value.detach().data.cpu().numpy() for key, value in loss.items()}
        return total_loss, loss
