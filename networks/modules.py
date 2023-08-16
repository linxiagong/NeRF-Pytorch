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
            **kwargs,
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

        self.input_dim = input_dim
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


class Embedder(nn.Module):
    def __init__(self,
                 input_dims,
                 include_input,
                 multires,
                 log_sampling,
                 periodic_fns=[torch.sin, torch.cos],
                 **kwargs):
        super(Embedder, self).__init__()
        self.input_dims = input_dims
        self.include_input = include_input
        self.max_freq_log2 = multires - 1
        self.num_freqs = multires
        self.log_sampling = log_sampling
        self.periodic_fns = periodic_fns
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.input_dims
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        if self.log_sampling:
            freq_bands = 2.**torch.linspace(0., self.max_freq_log2, steps=self.num_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**self.max_freq_log2, steps=self.num_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

    def forward(self, inputs):
        return self.embed(inputs)
