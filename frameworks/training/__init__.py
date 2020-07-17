from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__all__ = [
    'get_optimizer_strategy',
    'CameraClsTrainer',
    'CameraClsTrainer',
    'CamDataParallel'
]

from .optimizers import get_optimizer_strategy
from .trainer import CameraClsTrainer, CameraClsTrainer
from .data_parallel import CamDataParallel