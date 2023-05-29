"""Basic modules to build networks."""
from typing import Tuple

import torch
import torch.nn as nn

from .torch_utils import persistence


@persistence.persistent_class
class MLP(nn.Module):
    """Basic MLP class with hidden layers and an output layers."""
    def __init__(
            self,
            input_dim: int,
            depth: int,
            width: int,
            hidden_init=nn.init.xavier_uniform,
            hidden_activation=nn.ReLU,
            output_init=None,
            output_channels: int = 0,
            output_activation=None,
            use_bias: bool = True,
            skips: Tuple[int] = tuple(),
    ) -> None:
        super().__init__()

        layers = []
        for i in range(depth):
            in_channels = input_dim if i == 0 else width
            if i in skips:
                in_channels += input_dim
            layer = nn.Linear(in_channels, width, bias=use_bias)
            # init layer
            hidden_init(layer.weight)
            layer.bias.data.zero_()

            layers += [layer]
        self.mlp = nn.ModuleList(layers)
        self.hidden_activation = hidden_activation
        self.skips = skips

        self.output_channels = output_channels
        if output_channels > 0:
            self.output_linear = nn.Linear(width, output_channels, bias=use_bias)
            if output_init:
                output_init(self.output_linear.weight)
            self.output_linear.bias.data.zero_()
            self.output_activation = output_activation

        self.output_dim = output_channels or width

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.mlp):
            if i in self.skips:
                h = torch.cat([h, x], -1)
            h = self.mlp[i](h)
            h = self.hidden_activation()(h)

        if self.output_channels > 0:
            h = self.output_linear(h)
            if self.output_activation:
                h = self.output_activation()(h)
        return h
