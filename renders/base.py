from abc import ABC, abstractmethod
import torch.nn as nn


class BaseRender(ABC, nn.Module):
    """Serve a class; define necessary functions;"""
    @abstractmethod
    def sample_rays(self):
        """sample 3D points along the rays"""

    @abstractmethod
    def render(self):
        """render """

    def density(self, pts):
        """query sigma for points"""
        raise NotImplementedError
    
    def activate_density(self, density, interval):
        """compute alpha when given density and interval"""
        raise NotImplementedError