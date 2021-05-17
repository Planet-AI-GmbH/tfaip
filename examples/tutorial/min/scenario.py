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
from dataclasses import dataclass

from paiargparse import pai_dataclass

from tfaip import ScenarioBaseParams
from tfaip.imports import ScenarioBase
from examples.tutorial.min.data import TutorialTrainerGeneratorParams, TutorialDataParams
from examples.tutorial.min.model import TutorialModelParams


@pai_dataclass
@dataclass
class TutorialScenarioParams(ScenarioBaseParams[TutorialDataParams, TutorialModelParams]):
    pass


class TutorialScenario(ScenarioBase[TutorialScenarioParams, TutorialTrainerGeneratorParams]):
    pass
