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
"""Definition of the ConstantSchedule"""
from tfaip.trainer.scheduler import Constant
from tfaip.trainer.scheduler.learningrate import LearningRateSchedule


class ConstantSchedule(LearningRateSchedule):
    def __init__(self, params: Constant):
        super().__init__(params, "constant")

    def lr(self, epoch):
        return self.params.lr
