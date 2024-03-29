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
from dataclasses import dataclass, field

from paiargparse import pai_dataclass, pai_meta

from tfaip import LearningRateParams


@pai_dataclass(alt="WarmupDecay")
@dataclass
class WarmupDecayParams(LearningRateParams):
    """Cosine decay with warmup"""

    @staticmethod
    def cls():
        from tfaip.trainer.scheduler.warmup_decay import (
            WarmupDecaySchedule,
        )  # pylint: disable=import-outside-toplevel

        return WarmupDecaySchedule

    warmup_epochs: int = field(default=-1, metadata=pai_meta(help="Number of epochs for linear increase"))
    warmup_steps: int = field(default=-1, metadata=pai_meta(help="Number of epochs for linear increase"))
