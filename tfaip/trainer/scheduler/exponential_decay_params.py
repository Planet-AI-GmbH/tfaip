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
"""Definition of the ExponentialDecayParams"""
from dataclasses import dataclass, field

from paiargparse import pai_dataclass, pai_meta

from tfaip.trainer.scheduler.learningrate_params import LearningRateParams


@pai_dataclass(alt="ExponentialDecay")
@dataclass
class ExponentialDecayParams(LearningRateParams):
    """Exponential decay parameters"""

    @staticmethod
    def cls():
        from tfaip.trainer.scheduler.exponential_decay import (
            ExponentialDecaySchedule,
        )  # pylint: disable=import-outside-toplevel

        return ExponentialDecaySchedule

    learning_circle: int = field(
        default=3, metadata=pai_meta(help="(type dependent) The number of epochs with a flat constant learning rate")
    )
    lr_decay_rate: float = field(default=0.99, metadata=pai_meta(help="(type dependent) The exponential decay factor"))
