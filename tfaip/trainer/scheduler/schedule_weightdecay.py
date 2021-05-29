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
"""Definition of a WeightDecaySchedule"""

from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class WeightDecaySchedule(LearningRateSchedule):
    """The WeightDecaySchedule computes the weight decay multiplied by a rate

    Wrap the LR schedule and multiply by weight decay. Take care of base LR and modify weight decay accordingly
    """

    def __init__(self, weight_decay: float, learning_rate_schedule: "LearningRateSchedule", name=None, **kwargs):
        super().__init__(**kwargs)
        if name is None:
            name = self.__class__.__name__

        self.weight_decay = weight_decay / max(learning_rate_schedule.params.lr, 1e-8)
        self.learning_rate_schedule = learning_rate_schedule
        self.name = name

    def get_config(self):
        raise NotImplementedError  # should never be called, but is abstract

    def __call__(self, step):
        return self.weight_decay * self.learning_rate_schedule(step)
