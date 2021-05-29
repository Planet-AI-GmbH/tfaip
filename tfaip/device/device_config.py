# Copyright 2021 The tfaip authors. All Rights Reserved.
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
"""Implementation of the DeviceConfig and DeviceConfigParams

The device config sets up the GPUs to use for training.
"""
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

from paiargparse import pai_meta, pai_dataclass

from tfaip.util.enum import StrEnum

logger = logging.getLogger(__name__)


class DistributionStrategy(StrEnum):
    DEFAULT = "default"
    CENTRAL_STORAGE = "central_storage"
    MIRROR = "mirror"


def default_gpus():
    # if the env var CUDA_VISIBLE_DEVICES is set, use this as default, else an empty list
    devs = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if devs is None or len(devs) == 0:
        return []
    return list(range(len(devs.split(","))))  # tensorflow only 'sees' the visible devices, therefore just count them


@pai_dataclass
@dataclass
class DeviceConfigParams:
    """Configuration of the devices (GPUs).

    By default no gpys are added.
    Specify which gpus to use either by setting gpus or CUDA_VISIBLE_DEVICES
    """

    gpus: Optional[List[int]] = field(default=None, metadata=pai_meta(help="List of the GPUs to use."))
    gpu_auto_tune: bool = field(default=False, metadata=pai_meta(help="Enable auto tuning of the GPUs"))
    gpu_memory: Optional[int] = field(
        default=None,
        metadata=pai_meta(help="Limit the per GPU memory in MB. By default the memory will grow automatically"),
    )
    soft_device_placement: bool = field(default=True, metadata=pai_meta(help="Set up soft device placement is enabled"))
    dist_strategy: DistributionStrategy = field(
        default=DistributionStrategy.DEFAULT,
        metadata=pai_meta(help="Distribution strategy for multi GPU, select 'mirror' or 'central_storage'"),
    )


class DeviceConfig:
    """Manage the device config of Tensorflow.

    Usage (in a class, e.g. Trainer or LAV):
        - Add a member variable: `self.device_config = DeviceConfig(params)`
        - Wrap a function with `@distribute_strategy` to apply a Distribution strategy (optional)

    Note, DeviceConfig() must be called before any interaction (except static) with Tensorflow.
    Else, Tensorflow will use a default config which mismatches with this config which will result in an exception.
    """

    def __init__(self, params: DeviceConfigParams):
        import tensorflow as tf  # pylint: disable=import-outside-toplevel

        if params.dist_strategy is None:
            params.dist_strategy = DistributionStrategy.DEFAULT

        logger.info(f"Setting up device config {params}")
        self._params = params
        gpus = params.gpus if params.gpus is not None else default_gpus()

        os.environ["TF_CUDNN_USE_AUTOTUNE"] = "1" if self._params.gpu_auto_tune else "0"
        tf.config.set_soft_device_placement(self._params.soft_device_placement)
        physical_gpu_devices = tf.config.list_physical_devices("GPU")
        try:
            physical_gpu_devices = [physical_gpu_devices[i] for i in gpus]
        except IndexError as e:
            raise IndexError(
                f"GPU device not available. Number of devices detected: {len(physical_gpu_devices)}"
            ) from e

        tf.config.experimental.set_visible_devices(physical_gpu_devices, "GPU")
        for physical_gpu_device in physical_gpu_devices:
            tf.config.experimental.set_memory_growth(physical_gpu_device, self._params.gpu_memory is None)
            if self._params.gpu_memory is not None:
                tf.config.experimental.set_memory_growth(physical_gpu_device, False)
                tf.config.set_logical_device_configuration(
                    physical_gpu_device, [tf.config.LogicalDeviceConfiguration(memory_limit=self._params.gpu_memory)]
                )

        physical_gpu_device_names = ["/gpu:" + d.name.split(":")[-1] for d in physical_gpu_devices]
        logger.debug(f"Selecting strategy {self._params.dist_strategy.value}")

        # Select the distribution strategy
        if self._params.dist_strategy == DistributionStrategy.DEFAULT:
            self.strategy = None
        elif self._params.dist_strategy == DistributionStrategy.CENTRAL_STORAGE:
            self.strategy = tf.distribute.experimental.CentralStorageStrategy(compute_devices=physical_gpu_device_names)
        elif self._params.dist_strategy == DistributionStrategy.MIRROR:
            self.strategy = tf.distribute.MirroredStrategy(devices=physical_gpu_device_names)
        else:
            raise ValueError(f"Unknown strategy '{self._params.dist_strategy}'. Use 'mirror', 'central_storage'")


def distribute_strategy(func):
    # Wrap a function to be wrapped by the distribution strategy.
    # The owner (class) of the function must provide device_config (e.g. `self.device_config = DeviceConfig()`)
    def wrapper(*args, **kwargs):
        self = args[0]
        if not hasattr(self, "device_config"):
            raise AttributeError("To use distribute strategy, the class must provide a field 'device_config'")
        if self.device_config.strategy:
            with self.device_config.strategy.scope():
                return func(*args, **kwargs)
        return func(*args, **kwargs)

    return wrapper
