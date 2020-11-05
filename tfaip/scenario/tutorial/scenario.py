# Copyright 2020 The tfaip authors. All Rights Reserved.
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
from typing import Type

from tfaip.base.data.data import DataBase
from tfaip.base.scenario import ScenarioBase, ScenarioBaseParams
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from tfaip.base.model import ModelBase
from tfaip.scenario.tutorial.data import Data
from tfaip.scenario.tutorial.model import TutorialModel


@dataclass_json
@dataclass
class ScenarioParams(ScenarioBaseParams):
    pass


class TutorialScenario(ScenarioBase):
    @classmethod
    def model_cls(cls) -> Type['ModelBase']:
        return TutorialModel

    @classmethod
    def data_cls(cls) -> Type['DataBase']:
        return Data

    @staticmethod
    def get_params_cls() -> Type[ScenarioBaseParams]:
        return ScenarioParams

    def __init__(self, params: ScenarioParams):
        super().__init__(params)
