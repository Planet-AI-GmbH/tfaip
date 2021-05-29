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
"""Definition of the base LearningRateSchedule"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule as LearningRateScheduleBase

if TYPE_CHECKING:
    from tfaip import LearningRateParams


class LearningRateSchedule(LearningRateScheduleBase, ABC):
    """
    Base class for learning rate schedules.

    A custom implementation must overwrite lr(epoch)
    """

    def __init__(self, params: "LearningRateParams", name=None, **kwargs):
        super().__init__(**kwargs)
        if name is None:
            name = self.__class__.__name__
        self.params = params
        self.name = name
        assert self.params.steps_per_epoch > 0
        assert self.params.epochs > 0

    def get_config(self):
        return {"params": self.params.to_dict()}

    @classmethod
    def from_config(cls, config):
        from tfaip import LearningRateParams  # pylint: disable=import-outside-toplevel

        config["params"] = LearningRateParams.from_dict(config["params"])
        return cls(**config)

    def __call__(self, step):
        epoch = step // self.params.steps_per_epoch if self.params.step_function else step / self.params.steps_per_epoch
        epoch = tf.maximum(0.0, tf.cast(epoch - self.params.offset_epochs, tf.float32))
        return self.lr(epoch)

    @abstractmethod
    def lr(self, epoch) -> float:
        raise NotImplementedError
