# Copyright 2020 The tfaip authors. All Rights Reserved.
#
# This file is part of tfaip.
#
# tfaip is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# tfaip is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# tfaip. If not, see http://www.gnu.org/licenses/.
# ==============================================================================
from dataclasses import dataclass, field
from typing import List, Optional
import os
import logging

from dataclasses_json import dataclass_json

from tfaip.util.argument_parser import dc_meta
from tfaip.util.enum import StrEnum

logger = logging.getLogger(__name__)


class DistributionStrategy(StrEnum):
    Default = 'default'
    CentralStorage = 'central_storage'
    Mirror = 'mirror'


@dataclass_json
@dataclass
class DeviceConfigParams:
    gpus: List[int] = field(default_factory=list, metadata=dc_meta(
        help="List of the GPUs to use."
    ))
    gpu_auto_tune: bool = field(default=False, metadata=dc_meta(
        help="Enable auto tuning of the GPUs"
    ))
    gpu_memory: Optional[int] = field(default=None, metadata=dc_meta(
        help="Limit the per GPU memory in MB. By default the memory will grow automatically"
    ))
    soft_device_placement: bool = field(default=True, metadata=dc_meta(
        help="Set up soft device placement is enabled"
    ))
    dist_strategy: DistributionStrategy = field(default=DistributionStrategy.Default, metadata=dc_meta(
        help="Distribution strategy for multi GPU, select 'mirror' or 'central_storage'"
    ))


class DeviceConfig:
    def __init__(self, params: DeviceConfigParams):
        import tensorflow as tf

        logger.info("Setting up device config {}".format(params))
        self._params = params

        # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, self._params.gpus))
        os.environ['TF_CUDNN_USE_AUTOTUNE'] = '1' if self._params.gpu_auto_tune else '0'
        tf.config.set_soft_device_placement(self._params.soft_device_placement)
        physical_gpu_devices = tf.config.list_physical_devices('GPU')
        try:
            physical_gpu_devices = [physical_gpu_devices[i] for i in self._params.gpus]
        except IndexError:
            raise IndexError(f"GPU device not available. Number of devices detected: {len(physical_gpu_devices)}")

        tf.config.experimental.set_visible_devices(physical_gpu_devices, 'GPU')
        for physical_gpu_device in physical_gpu_devices:
            tf.config.experimental.set_memory_growth(physical_gpu_device, self._params.gpu_memory is None)
            if self._params.gpu_memory is not None:
                tf.config.experimental.set_memory_growth(physical_gpu_device, False)
                tf.config.set_logical_device_configuration(physical_gpu_device, [
                    tf.config.LogicalDeviceConfiguration(memory_limit=self._params.gpu_memory)])

        physical_gpu_device_names = ['/gpu:' + d.name.split(':')[-1] for d in physical_gpu_devices]
        logger.debug(f"Selecting strategy {self._params.dist_strategy.value}")
        if self._params.dist_strategy == DistributionStrategy.Default:
            self.strategy = None
        elif self._params.dist_strategy == DistributionStrategy.CentralStorage:
            self.strategy = tf.distribute.experimental.CentralStorageStrategy(compute_devices=physical_gpu_device_names)
        elif self._params.dist_strategy == DistributionStrategy.Mirror:
            self.strategy = tf.distribute.MirroredStrategy(devices=physical_gpu_device_names)
        else:
            raise ValueError(f"Unknown strategy '{self._params.dist_strategy}'. Use 'mirror', 'central_storage'")


def distribute_strategy(train):
    def wrapper(*args, **kwargs):
        trainer_or_lav = args[0]
        if trainer_or_lav.device_config.strategy:
            with trainer_or_lav.device_config.strategy.scope():
                return train(*args, **kwargs)
        return train(*args, **kwargs)

    return wrapper
