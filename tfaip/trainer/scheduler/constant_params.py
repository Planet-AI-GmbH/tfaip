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
"""Definition of the Constant learning rate schedule"""
from dataclasses import dataclass

from paiargparse import pai_dataclass

from tfaip.trainer.scheduler.learningrate_params import LearningRateParams


@pai_dataclass
@dataclass
class Constant(LearningRateParams):
    @staticmethod
    def cls():
        from tfaip.trainer.scheduler.constant import ConstantSchedule  # pylint: disable=import-outside-toplevel

        return ConstantSchedule
