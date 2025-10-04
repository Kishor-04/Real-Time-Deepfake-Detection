"""Model architectures for deepfake detection"""

from .xception_model import XceptionDeepfakeDetector, load_pretrained_xception
from .efficientnet_model import EfficientNetDeepfakeDetector, load_pretrained_efficientnet

__all__ = [
    'XceptionDeepfakeDetector',
    'load_pretrained_xception',
    'EfficientNetDeepfakeDetector',
    'load_pretrained_efficientnet'
]