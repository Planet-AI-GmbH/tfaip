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
"""Definition of the CosineDecayParams, WarmupCosineDecayParams, and WarmupConstantCosineDecayParams"""
from dataclasses import dataclass, field

from paiargparse import pai_dataclass, pai_meta

from tfaip.trainer.scheduler.learningrate_params import LearningRateParams


@pai_dataclass(alt="CosineDecay")
@dataclass
class CosineDecayParams(LearningRateParams):
    """Definition of a cosine decay"""

    @staticmethod
    def cls():
        from tfaip.trainer.scheduler.cosine_decay import CosineDecaySchedule  # pylint: disable=import-outside-toplevel

        return CosineDecaySchedule

    learning_circle: int = field(
        default=3, metadata=pai_meta(help="(type dependent) The number of epochs with a flat constant learning rate")
    )
    lr_decay_rate: float = field(default=0.99, metadata=pai_meta(help="(type dependent) The exponential decay factor"))
    decay_fraction: float = field(default=0.1, metadata=pai_meta(help="(type dependent) Alpha value of cosine decay"))
    final_epochs: int = field(
        default=50,
        metadata=pai_meta(help="(type dependent) Number of final epochs with a steep decline in the learning rate"),
    )


@pai_dataclass(alt="WarmupConstantDecay")
@dataclass
class WarmupCosineDecayParams(CosineDecayParams):
    """Cosine decay with warmup"""

    @staticmethod
    def cls():
        from tfaip.trainer.scheduler.cosine_decay import (
            WarmupCosineDecaySchedule,
        )  # pylint: disable=import-outside-toplevel

        return WarmupCosineDecaySchedule

    warmup_epochs: int = field(
        default=10, metadata=pai_meta(help="(type dependent) Number of epochs with an increasing learning rate.")
    )
    warmup_factor: int = field(
        default=10, metadata=pai_meta(help="(type dependent) Factor from which to start warmup learning (lr/fac)")
    )


@pai_dataclass(alt="WarmupConstantCosineDecay")
@dataclass
class WarmupConstantCosineDecayParams(WarmupCosineDecayParams):
    @staticmethod
    def cls():
        from tfaip.trainer.scheduler.cosine_decay import (
            WarmupConstantCosineDecaySchedule,
        )  # pylint: disable=import-outside-toplevel

        return WarmupConstantCosineDecaySchedule

    constant_epochs: int = field(
        default=10, metadata=pai_meta(help="(Type dependent) Number of constant epochs before starting lr decay")
    )
