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
"""Definition of GraphBase"""
from typing import TypeVar

from tfaip import ModelBaseParams
from tfaip.model.layerbase import LayerBase
from tfaip.util.enum import StrEnum

TGP = TypeVar('TGP', bound=ModelBaseParams)


class KeyOutput(StrEnum):
    PRED = 'pred'  # prediction which are values in [0,1] for each class-dimension. Last dimension is class dimension
    CLASS = 'class'  # classification which is the argmax of 'pred'
    LOGITS = 'logits'  # logits of classifications before applying softmax or sigmoid


class GraphBase(LayerBase[TGP]):
    pass
