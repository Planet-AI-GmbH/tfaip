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
from typing import Type
from dataclasses import dataclass

from paiargparse import pai_dataclass
from tfaip import DataBaseParams, TrainerPipelineParams

from examples.atr.datapipeline.datagenerator import ATRDataGeneratorParams

this_dir = os.path.dirname(os.path.realpath(__file__))


class Keys:
    Targets = "targets"
    TargetsLength = "targets_length"
    Image = "image"
    ImageLength = "imageLength"


def read_default_codec() -> str:
    with open(os.path.join(this_dir, "workingdir", "uw3_50lines", "full_codec.txt")) as f:
        return f.read()


@pai_dataclass
@dataclass
class ATRDataParams(DataBaseParams):
    codec: str = read_default_codec()
    height: int = 48

    @staticmethod
    def cls():
        from examples.atr.data import ATRData

        return ATRData


# Setup the training pipeline to use both ATRDataGeneratorParams for the training- and validation-set
@pai_dataclass
@dataclass
class ATRTrainerPipelineParams(TrainerPipelineParams[ATRDataGeneratorParams, ATRDataGeneratorParams]):
    pass
