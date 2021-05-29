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
"""Definition of a ListFileScenario"""
from tfaip.scenario.listfile.listfilelav import ListFileLAV
from tfaip.scenario.listfile.params import ListFileTrainerPipelineParams
from tfaip.scenario.scenariobase import ScenarioBase, TScenarioParams


class ListFileScenario(ScenarioBase[TScenarioParams, ListFileTrainerPipelineParams]):
    """
    Base-Class for a Scenario working with list files.
    A list file is a simple text file where each line is the path to a sample.
    The ListFileScenario uses ListFileTrainerPipelineParams which create a DataGenerator that will yield each line
    as a Sample's input and target. This must then be processed by DataProcessors.
    """

    @classmethod
    def lav_cls(cls):
        return ListFileLAV
