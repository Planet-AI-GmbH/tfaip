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
from typing import Type, Dict

import numpy as np
from paiargparse import pai_dataclass

from examples.tutorial.full.data.data import TutorialDataParams
from examples.tutorial.full.data.prediction_data_generation import TutorialPredictionGeneratorParams
from examples.tutorial.full.data.training_data_generation import TutorialTrainerGeneratorParams
from examples.tutorial.full.model import TutorialModelParams
from examples.tutorial.full.predictor import TutorialMultiModelPredictor
from tfaip import Sample
from tfaip.imports import EvaluatorBase, ScenarioBase, ScenarioBaseParams, MultiModelPredictor
from tfaip.util.typing import AnyNumpy


@pai_dataclass
@dataclass
class TutorialScenarioParams(ScenarioBaseParams[TutorialDataParams, TutorialModelParams]):
    pass


class TutorialScenario(ScenarioBase[TutorialScenarioParams, TutorialTrainerGeneratorParams]):
    @classmethod
    def predict_generator_params_cls(cls):
        return TutorialPredictionGeneratorParams

    @classmethod
    def multi_predictor_cls(cls) -> Type["MultiModelPredictor"]:
        return TutorialMultiModelPredictor

    @classmethod
    def evaluator_cls(cls) -> Type["EvaluatorBase"]:
        class MNISTEvaluator(EvaluatorBase):
            def __init__(self, params):
                super(MNISTEvaluator, self).__init__(params)
                self.true_count = 0
                self.total_count = 0

            def __enter__(self):
                self.true_count = 0
                self.total_count = 0

            def update_state(self, sample: Sample):
                self.total_count += 1
                self.true_count += np.sum(sample.targets["gt"] == sample.outputs["class"])

            def result(self) -> Dict[str, AnyNumpy]:
                return {"eval_acc": self.true_count / self.total_count if self.total_count else 0}

        return MNISTEvaluator
