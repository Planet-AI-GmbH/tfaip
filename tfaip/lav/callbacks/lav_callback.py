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
"""Implementation of a general LAVCallback"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from tfaip import DataGeneratorParams

if TYPE_CHECKING:
    from tfaip.lav.lav import LAV
    from tfaip.model.modelbase import ModelBase
    from tfaip.data.data import DataBase


class LAVCallback(ABC):
    """
    Base class of a callback that can be added to LAV
    """

    def __init__(self):
        self.lav: "LAV" = None  # Set from lav
        self.data: "DataBase" = None  # Set from lav
        self.model: "ModelBase" = None  # set from lav

    @abstractmethod
    def on_sample_end(self, data_generator_params: DataGeneratorParams, sample):
        pass

    @abstractmethod
    def on_lav_end(self, data_generator_params: DataGeneratorParams, result):
        pass
