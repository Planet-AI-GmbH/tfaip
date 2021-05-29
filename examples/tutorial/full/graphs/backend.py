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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Generic

from paiargparse import pai_dataclass
from tensorflow import keras
from tfaip.model.layerbase import TFAIPLayerBase


@pai_dataclass
@dataclass
class TutorialBackendParams(ABC):
    @abstractmethod
    def cls(self) -> "TutorialBackend":
        raise NotImplementedError


TBP = TypeVar("TBP", bound=TutorialBackendParams)


class TutorialBackend(Generic[TBP], TFAIPLayerBase):
    """
    This class defines the interface for creating backends
    """

    def __init__(self, params: TBP, **kwargs):
        super(TutorialBackend, self).__init__(**kwargs)
        self.params = params
