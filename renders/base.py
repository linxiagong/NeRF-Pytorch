from abc import ABC, abstractmethod

class BaseRender(ABC):
    @abstractmethod
    def sample_rays(self):
        """sample 3D points along the rays"""

    @abstractmethod
    def render(self):
        """render """