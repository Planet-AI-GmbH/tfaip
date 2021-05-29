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
"""Definition of the ExponentialDecaySchedule"""
from tensorflow import keras

from tfaip.trainer.scheduler import ExponentialDecayParams
from tfaip.trainer.scheduler.learningrate import LearningRateSchedule


class ExponentialDecaySchedule(LearningRateSchedule):
    """Exponential decay

    This class simply wraps keras.optimizers.schedules.ExponentialDecay
    """

    def __init__(
        self,
        params: ExponentialDecayParams,
    ):
        super().__init__(params)
        self.exp_decay = keras.optimizers.schedules.ExponentialDecay(
            params.lr, params.learning_circle, params.lr_decay_rate, staircase=True, name=self.name
        )

    def lr(self, epoch):
        return self.exp_decay(epoch)
