"""
DVGO Implementation
Reference: https://github.com/sunset1995/DirectVoxGO
Difference with NeRFRender:
    - DO NOT sample the same amound of points on each ray
        |-> adjusted funcs: sample_rays, raw2alpha
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .nerf_render import NeRFRender

class DVGORender(NeRFRender):
    def __init__(self, network: nn.Module, **render_kwargs):
        super().__init__(network, **render_kwargs)