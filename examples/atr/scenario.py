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
from dataclasses import dataclass

from paiargparse import pai_dataclass
from tfaip import ScenarioBaseParams
from tfaip.scenario.scenariobase import ScenarioBase

from examples.atr.model import ATRModelParams
from examples.atr.params import ATRDataParams, ATRTrainerPipelineParams

this_dir = os.path.dirname(os.path.realpath(__file__))


@pai_dataclass
@dataclass
class ATRScenarioParams(ScenarioBaseParams[ATRDataParams, ATRModelParams]):
    def __post_init__(self):
        self.model.num_classes = len(self.data.codec) + 1  # +1 for blank


class ATRScenario(ScenarioBase[ATRScenarioParams, ATRTrainerPipelineParams]):
    @classmethod
    def default_trainer_params(cls):
        p = super().default_trainer_params()
        default_data_dir = os.path.join(this_dir, "workingdir", "uw3_50lines")
        p.gen.train.image_files = [os.path.join(default_data_dir, "train", "*.png")]
        p.gen.val.image_files = [os.path.join(default_data_dir, "train", "*.png")]
        p.gen.setup.train.batch_size = 5
        p.gen.setup.val.batch_size = 5
        p.samples_per_epoch = 1024
        return p
