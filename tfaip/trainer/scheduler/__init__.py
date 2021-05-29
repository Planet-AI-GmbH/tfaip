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
"""Imports of all Schedules and definition of the DEFAULT_SCHEDULES"""
from tfaip.trainer.scheduler.constant_params import Constant
from tfaip.trainer.scheduler.cosine_decay_params import (
    WarmupCosineDecayParams,
    WarmupConstantCosineDecayParams,
    CosineDecayParams,
)
from tfaip.trainer.scheduler.exponential_decay_params import ExponentialDecayParams
from tfaip.trainer.scheduler.learningrate_params import LearningRateParams

DEFAULT_SCHEDULES = [
    WarmupCosineDecayParams,
    WarmupConstantCosineDecayParams,
    CosineDecayParams,
    Constant,
    ExponentialDecayParams,
]
