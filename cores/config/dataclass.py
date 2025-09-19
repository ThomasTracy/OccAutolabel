from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class CameraConfig:
    name: str
    type: str
    image_size: np.ndarray
    intrinsic_parameters: np.ndarray
    
    @property
    def width(self) -> int:
        return self.image_size[0]
    
    @property
    def height(self) -> int:
        return self.image_size[1]

@dataclass
class LidarConfig:
    """LiDAR配置"""
    name: str
    type: str
    channels: int
    horizontal_resolution: float
    vertical_resolution: float
    max_range: float
    min_range: float
    extrinsic_parameters: np.ndarray