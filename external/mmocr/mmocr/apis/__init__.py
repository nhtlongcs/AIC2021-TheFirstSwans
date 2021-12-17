# Copyright (c) OpenMMLab. All rights reserved.
from .inference import init_detector, model_inference
from .train import train_detector

__all__ = ['model_inference', 'train_detector', 'init_detector']
