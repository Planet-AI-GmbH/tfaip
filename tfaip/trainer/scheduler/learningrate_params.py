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
"""Definition of the LearningRateParams"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from paiargparse import pai_meta, pai_dataclass


@pai_dataclass
@dataclass
class LearningRateParams(ABC):
    """Base definition of learning rate params"""

    @staticmethod
    @abstractmethod
    def cls():
        raise NotImplementedError

    def create(self):
        if self.epochs < 0:
            raise ValueError("Epochs not specified.")

        return self.cls()(self)

    lr: float = field(default=0.001, metadata=pai_meta(help="The learning rate."))
    step_function: bool = field(
        default=True, metadata=pai_meta(help="(type dependent) Step function of exponential decay.")
    )
    offset_epochs: int = field(
        default=0,
        metadata=pai_meta(
            help="Offset to subtract from the current training epoch (if the total is negative it will be capped at 0, "
            "and i.e., if < 0 the total epoch is greater than the training epoch). "
            "Can be used to reset the learning rate schedule when resuming training."
        ),
    )

    # Updated during training to support loading and resuming
    steps_per_epoch: int = field(default=-1, metadata=pai_meta(mode="ignore"))
    epochs: int = field(default=-1, metadata=pai_meta(mode="ignore"))
