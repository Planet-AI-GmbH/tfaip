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
"""Definition of the parameters for a ListFileScenario"""
from dataclasses import dataclass, field
from typing import Optional, List

from paiargparse import pai_meta, pai_dataclass

from tfaip import TrainerPipelineParams
from tfaip.data.databaseparams import DataGeneratorParams


@pai_dataclass
@dataclass
class ListsFileGeneratorParams(DataGeneratorParams):
    """
    Parameters for the ListsFileDataGenerator
    """

    @staticmethod
    def cls():
        from tfaip.scenario.listfile.datagenerator import (
            ListsFileDataGenerator,
        )  # pylint: disable=import-outside-toplevel

        return ListsFileDataGenerator

    lists: Optional[List[str]] = field(default_factory=list, metadata=pai_meta(help="Training list files."))
    list_ratios: Optional[List[float]] = field(
        default=None, metadata=pai_meta(help="Ratios of picking list files. Must be supported by the scenario")
    )

    def __post_init__(self):
        if self.lists:
            if not self.list_ratios:
                self.list_ratios = [1.0] * len(self.lists)
            else:
                if len(self.list_ratios) != len(self.lists):
                    raise ValueError(
                        f"Length of list_ratios must be equals to number of lists. "
                        f"Got {self.list_ratios}!={self.lists}"
                    )


@pai_dataclass
@dataclass
class ListFileTrainerPipelineParams(TrainerPipelineParams[ListsFileGeneratorParams, ListsFileGeneratorParams]):
    """
    Implemented TrainerPipelineParams that replaces the defaults of train and val to ListFileGeneratorParams.
    """

    pass
