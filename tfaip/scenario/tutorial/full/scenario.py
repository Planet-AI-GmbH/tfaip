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
from typing import Type, Dict
import numpy as np
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from tfaip.base.data.data import DataBase
from tfaip.base.data.pipeline.definitions import Sample
from tfaip.base.evaluator.evaluator import Evaluator
from tfaip.base.imports import ScenarioBase, ScenarioBaseParams, ModelBase
from tfaip.scenario.tutorial.full.model import TutorialModel
from tfaip.scenario.tutorial.full.predictor import TutorialMultiModelPredictor
from tfaip.util.typing import AnyNumpy


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
        from tfaip.scenario.tutorial.full.data.data import Data
        return Data

    @staticmethod
    def get_params_cls() -> Type[ScenarioBaseParams]:
        return ScenarioParams

    @classmethod
    def multi_predictor_cls(cls) -> Type['MultiModelPredictor']:
        return TutorialMultiModelPredictor

    @classmethod
    def evaluator_cls(cls) -> Type['Evaluator']:
        class MNISTEvaluator(Evaluator):
            def __init__(self, params):
                super(MNISTEvaluator, self).__init__(params)
                self.true_count = 0
                self.total_count = 0

            def __enter__(self):
                self.true_count = 0
                self.total_count = 0

            def update_state(self, sample: Sample):
                self.total_count += 1
                self.true_count += np.sum(sample.targets['gt'] == sample.outputs['class'])

            def result(self) -> Dict[str, AnyNumpy]:
                return {'eval_acc': self.true_count / self.total_count}

        return MNISTEvaluator

    def __init__(self, params: ScenarioParams):
        super().__init__(params)
