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
import os
from dataclasses import dataclass, field
from typing import List, Type

from paiargparse import pai_dataclass
from tfaip import DataBaseParams

this_dir = os.path.dirname(os.path.realpath(__file__))


class Keys:
    Target = "target"
    Image = "image"
    OutputLogits = "logits"
    OutputSoftmax = "softmax"
    OutputClass = "class"
    OutputClassName = "class_name"


@pai_dataclass
@dataclass
class ICDataParams(DataBaseParams):
    @staticmethod
    def cls():
        from examples.imageclassification.data import ICData

        return ICData

    classes: List[str] = field(default_factory=list)
    image_height: int = 180
    image_width: int = 180
