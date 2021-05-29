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

from examples.template.general.datapipeline.datagenerator import TemplateDataGeneratorParams
from tfaip import ScenarioBaseParams, DataBaseParams, ModelBaseParams, TrainerPipelineParams


@pai_dataclass
@dataclass
class TemplateTrainerGeneratorParams(TrainerPipelineParams[TemplateDataGeneratorParams, TemplateDataGeneratorParams]):
    # Implementing TrainerPipelineParams will add two distinct data generators for training and validation
    # Inherit TrainerPipelineParamsBase for more freedom.
    pass


@pai_dataclass
@dataclass
class TemplateDataParams(DataBaseParams):
    # [Add global data params that can be accessed by every data processor]

    @staticmethod
    def cls():
        from examples.template.general.data import TemplateData

        return TemplateData


@pai_dataclass
@dataclass
class TemplateModelParams(ModelBaseParams):
    # [Add general model params, also including parameters that define the graph]

    @staticmethod
    def cls():
        from examples.template.general.model import TemplateModel

        return TemplateModel

    def graph_cls(self):
        from examples.template.general.graphs import TemplateGraph

        return TemplateGraph


@pai_dataclass
@dataclass
class TemplateScenarioParams(ScenarioBaseParams[TemplateDataParams, TemplateModelParams]):
    # [Usually no global scenario params are required]
    pass
