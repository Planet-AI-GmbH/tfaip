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
"""Implementation of MultiLAV"""
import json
from abc import ABC
from typing import Type, List, Callable

from tfaip import LAVParams, DataGeneratorParams, PredictorParams
from tfaip import Sample
from tfaip.device.device_config import DeviceConfig, distribute_strategy
from tfaip.evaluator.evaluator import EvaluatorBase
from tfaip.lav.callbacks.lav_callback import LAVCallback
from tfaip.predict.multimodelpredictor import MultiModelPredictor
from tfaip.trainer.callbacks.benchmark_callback import BenchmarkResults


class MultiLAV(ABC):
    """
    Similar to LAV but supports multiple models and voting.
    Drawback: does not use metrics defined in model (EvaluatorBase and Voter must handle everything)

    Therefore the implementation differs from the classical LAV, a MultiPredictor is created whose outputs are passed
    to the evaluator.
    """

    @classmethod
    def params_cls(cls) -> Type[LAVParams]:
        return LAVParams

    def __init__(
        self,
        params: LAVParams,
        predictor_fn: Callable[[List[str], PredictorParams], MultiModelPredictor],
        evaluator: EvaluatorBase,
        predictor_params: PredictorParams,
    ):
        assert params.model_path
        self._params = params
        self._predictor_fn = predictor_fn
        self._evaluator = evaluator
        self.device_config = DeviceConfig(self._params.device)
        self.benchmark_results = BenchmarkResults()
        self.predictor_params = predictor_params
        self.predictor_params.pipeline = self._params.pipeline
        predictor_params.silent = True
        predictor_params.progress_bar = True
        predictor_params.include_targets = True

    @distribute_strategy
    def run(
        self,
        data_gen_params: DataGeneratorParams,
        callbacks: List[LAVCallback] = None,
        run_eagerly=False,
    ):
        if callbacks is None:
            callbacks = []

        self.predictor_params.run_eagerly = run_eagerly
        predictor = self._predictor_fn(self._params.model_path, self.predictor_params)
        lav_pipeline = predictor.data.get_or_create_pipeline(self._params.pipeline, data_gen_params)

        for cb in callbacks:
            cb.lav, cb.data, cb.model = self, predictor.data, None

        with self._evaluator:
            for prediction in predictor.predict_pipeline(lav_pipeline):
                targets, outputs = prediction.targets, prediction.outputs

                for cb in callbacks:
                    cb.on_sample_end(data_gen_params, prediction)

                if not self._params.silent:
                    print(targets, outputs)

                self._evaluator.update_state(prediction)

            result = self._evaluator.result()

            for cb in callbacks:
                cb.on_lav_end(result)

            if not self._params.silent:
                print(json.dumps(result, indent=2))

            self.benchmark_results = predictor.benchmark_results
            yield result

    def extract_dump_data(self, sample: Sample):
        return sample.targets, sample.outputs
