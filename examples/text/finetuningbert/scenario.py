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

from examples.text.finetuningbert.datapipeline.gluedata import GlueTrainerPipelineParams
from examples.text.finetuningbert.datapipeline.tokenizerprocessor import TokenizerProcessorParams
from examples.text.finetuningbert.model import FTBertModelParams
from examples.text.finetuningbert.params import FTBertDataParams
from tfaip import ScenarioBaseParams
from tfaip.scenario.scenariobase import ScenarioBase
from tfaip.trainer.scheduler import WarmupCosineDecayParams


@pai_dataclass
@dataclass
class FTBertScenarioParams(ScenarioBaseParams[FTBertDataParams, FTBertModelParams]):
    def __post_init__(self):
        for p in self.data.pre_proc.processors_of_type(TokenizerProcessorParams):
            p.model_name = self.model.model_name


class FTBertScenario(ScenarioBase[FTBertScenarioParams, GlueTrainerPipelineParams]):
    @classmethod
    def default_trainer_params(cls):
        p = super().default_trainer_params()
        p.gen.setup.train.batch_size = 32
        p.gen.setup.val.batch_size = 32
        p.epochs = 3
        p.learning_rate = WarmupCosineDecayParams(lr=2e-5, warmup_epochs=1, warmup_factor=100)
        return p
