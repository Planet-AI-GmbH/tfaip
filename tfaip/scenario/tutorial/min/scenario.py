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
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from tfaip.base.data.data import DataBase
from tfaip.base.imports import ScenarioBase, ScenarioBaseParams, ModelBase
from tfaip.scenario.tutorial.min.data import Data
from tfaip.scenario.tutorial.min.model import TutorialModel


@dataclass_json
@dataclass
class ScenarioParams(ScenarioBaseParams):
    # Optionally add parameters that are scenario specific and to not fit in the DataParams or ModelParams
    pass


class TutorialScenario(ScenarioBase):
    @classmethod
    def model_cls(cls) -> Type['ModelBase']:
        # Which model class belongs to this scenario
        return TutorialModel

    @classmethod
    def data_cls(cls) -> Type['DataBase']:
        # Which data class belongs to this scenario
        return Data

    @staticmethod
    def get_params_cls() -> Type[ScenarioBaseParams]:
        # Which parameter class belongs to this scenario
        return ScenarioParams
