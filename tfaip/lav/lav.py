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
"""Implementation of LAV"""
import json
import logging
import os
from abc import ABC
from typing import TYPE_CHECKING, Type, Dict, List, Callable, Optional, Iterable

import numpy as np
import tensorflow.keras as keras

from tfaip import LAVParams, DataGeneratorParams, PredictorParams
from tfaip import Sample
from tfaip.device.device_config import DeviceConfig, distribute_strategy
from tfaip.evaluator.evaluator import EvaluatorBase
from tfaip.lav.callbacks.lav_callback import LAVCallback
from tfaip.predict.predictor import Predictor
from tfaip.predict.predictorbase import PredictorBenchmarkResults
from tfaip.util.file.oshelper import ChDir
from tfaip.util.multiprocessing.parallelmap import tqdm_wrapper

if TYPE_CHECKING:
    from tfaip.data.data import DataBase
    from tfaip.model.modelbase import ModelBase

logger = logging.getLogger(__name__)


class MetricsAccumulator:
    """
    Computation of the running weighted sum (=weighted mean) of a dictionary of float (here the metrics of the scenario)
    Call final() to access the mean.

    Attributes:
        running_sum: The running weighted sum of all values
        running_weight: The sum of all weights
    """

    def __init__(self):
        self.running_sum: Dict[str, float] = {}
        self.running_weight: Dict[str, float] = {}

    def accumulate_dict_sum(self, new_values: Dict[str, float], sample_weights: Dict[str, float]):
        def weighted(value, weight):
            if weight is None:
                return value

            return value * weight

        if len(self.running_sum) == 0:
            self.running_sum = {k: 0 for k, v in new_values.items()}
            self.running_weight = {k: 0 for k, v in new_values.items()}

        self.running_sum = {k: (self.running_sum[k] + weighted(v, sample_weights.get(k, None))) for k, v in
                            new_values.items()}
        self.running_weight = {k: (v + (1 if k not in sample_weights else sample_weights[k])) for k, v in
                               self.running_weight.items()}

    def final(self):
        return {k: v / self.running_weight[k] for k, v in self.running_sum.items()}


class LAV(ABC):
    """
    This is the base Class for Loading And Validation (LAV) a scenario. By default, LAV is independent of the scenario
    and should therefore not be overridden (but it is possible though in Meta of a Scenario).

    LAV will open an exported (or the best) model, create the metrics of the actual scenario. Then predict is called
    on the model on the DataBase validation(!) set and fed to the metric to obtain results (MetricsAccumulator).
    During the prediction a models print_evaluate method is called to print informative text.

    The first possibility to override LAV is to inherit and use _on_batch_end, _on_sample_end, _on_lav_end with custom
    parameters. Thus is used for instance to support rendering of attention matrices in atr.transformer.
    """

    @classmethod
    def params_cls(cls) -> Type[LAVParams]:
        return LAVParams

    def __init__(self,
                 params: LAVParams,
                 data_fn: Callable[[], 'DataBase'],
                 model_fn: Callable[[], 'ModelBase'],
                 predictor_fn: Callable[[PredictorParams, 'DataBase'], Predictor],
                 evaluator_fn: Callable[[], EvaluatorBase],
                 ):
        assert params.model_path
        self._params = params
        self._data_fn = data_fn
        self._predictor_fn = predictor_fn
        self._model_fn = model_fn
        self._evaluator_fn = evaluator_fn
        self.device_config = DeviceConfig(self._params.device)
        self._data: Optional['DataBase'] = None
        self._model: Optional['ModelBase'] = None
        self.benchmark_results = PredictorBenchmarkResults()

    @distribute_strategy
    def run(self,
            generator_params: Iterable[DataGeneratorParams],
            model: keras.Model = None,
            run_eagerly=False,
            callbacks: List[LAVCallback] = None,
            ) -> Iterable[Dict[str, float]]:
        callbacks = callbacks if callbacks else []
        with ChDir(os.path.join(self._params.model_path)):
            # resources are located in parent dir
            self._data = self._data_fn()
        self._model = self._model_fn()
        predictor_params = PredictorParams(self._params.device, silent=True, progress_bar=True)
        predictor: Predictor = self._predictor_fn(predictor_params, self._data)
        evaluator: EvaluatorBase = self._evaluator_fn()

        for cb in callbacks:
            cb.lav, cb.data, cb.model = self, self._data, self._model

        if run_eagerly:
            logger.warning('Running in eager mode. Use this only for debugging, since the graph of the saved model '
                           'might get changed due to "reconstruction" of the graph')
            custom_objects = self._model.__class__.all_custom_objects()
        else:
            custom_objects = None

        keras_model: keras.Model = model
        if not model:
            keras_model = keras.models.load_model(os.path.join(self._params.model_path, 'serve'),
                                                  compile=False, custom_objects=custom_objects)

        # create a new keras model that uses the inputs and outputs of the loaded model but adds the targets of the
        # dataset. Then create the metrics as output of the new model
        real_inputs = keras_model.input
        real_targets = self._data.create_target_as_input_layers()
        real_meta = self._data.create_meta_as_input_layers()
        eval_inputs = {**real_inputs, **real_targets, **real_meta}
        sample_weights = {k + '_sample_weight': v for k, v in
                          self._model.sample_weights(real_inputs, real_targets).items()}
        metric_outputs = self._model.extended_metric(eval_inputs, keras_model.output)
        simple_metrics = self._model.metric()
        eval_model = keras.Model(eval_inputs,
                                 [{**real_inputs, **real_targets, **sample_weights},
                                  {**metric_outputs, **keras_model.output}, real_meta])
        if run_eagerly:
            eval_model._run_eagerly = True  # pylint: disable=protected-access

        # I know what I am doing: already wrapped inputs and targets
        predictor._keras_model = eval_model  # pylint: disable=protected-access

        def predict_pipeline(pipeline):
            with pipeline as rd:
                def regroup(i, t, m):
                    return {**i, **t, **m}, t

                # todo (christoph) if input is none warning/exception
                input_dataset = rd.input_dataset().map(regroup)

                def extract_metric(s: Sample):
                    i, o, m = s.inputs, s.outputs, s.meta
                    m = m or {}
                    # Add metrics as meta information
                    return Sample(inputs=i,
                                  outputs={k: v for k, v in o.items() if k not in metric_outputs},
                                  meta={**m, 'lav_metrics': {k: v for k, v in o.items() if k in metric_outputs}},
                                  )

                def ungroup(sample):
                    inputs = {k: v for k, v in sample.inputs.items() if k in real_inputs or k in sample_weights}
                    targets = {k: v for k, v in sample.inputs.items() if k in real_targets}
                    return sample.new_inputs(inputs).new_targets(targets)

                for r in tqdm_wrapper(
                        rd.process_output(map(ungroup, map(extract_metric, predictor.predict_dataset(input_dataset)))),
                        progress_bar=predictor_params.progress_bar,
                        desc='LAV',
                        total=len(rd),
                ):
                    yield r

        # accumulate the mean
        metrics_accum = MetricsAccumulator()
        for params in generator_params:
            val_data = self._data.create_pipeline(self._params.pipeline, params)
            with evaluator:
                for sample in predict_pipeline(val_data):
                    # unpack the packed sample and create a real sample without sample weights and the packed metrics
                    # sample weights and inputs are both mapped into sample.inputs
                    un_batched_inputs = {k: v for k, v in sample.inputs.items() if k in real_inputs}
                    un_batched_sample_weights = {k[:-14]: v for k, v in sample.inputs.items() if k in sample_weights}
                    un_batched_targets = sample.targets
                    un_batched_outputs = sample.outputs
                    un_batched_metric_outputs = sample.meta['lav_metrics']
                    un_batched_meta = sample.meta.copy()
                    del sample.meta['lav_metrics']
                    un_batched_sample = Sample(inputs=un_batched_inputs, outputs=un_batched_outputs,
                                               targets=un_batched_targets, meta=un_batched_meta)
                    if not self._params.silent:
                        self._model.print_evaluate(un_batched_sample, self._data)

                    for k, metric in simple_metrics.items():
                        # metrics expect a batch dimension, thus wrap into a list
                        metric.metric.update_state(
                            np.expand_dims(un_batched_targets[metric.target], axis=0),
                            np.expand_dims(un_batched_outputs[metric.output], axis=0),
                            np.expand_dims(un_batched_sample_weights[k],
                                           axis=0) if k in un_batched_sample_weights else None)

                    metrics_accum.accumulate_dict_sum(un_batched_metric_outputs, un_batched_sample_weights)

                    self._on_sample_end(params, un_batched_sample)
                    for cb in callbacks:
                        cb.on_sample_end(params, un_batched_sample)

                    evaluator.update_state(
                        Sample(outputs=un_batched_outputs, targets=un_batched_targets, meta=sample.meta))

            # print the output
            all_metric_results = {**metrics_accum.final(),
                                  **{k: float(v.metric.result().numpy()) for k, v in simple_metrics.items() if
                                     not self._model.tensorboard_handler.is_tensorboard_only(k, v.metric.result())},
                                  **{k: v.metric.result().numpy() for k, v in simple_metrics.items() if
                                     self._model.tensorboard_handler.is_tensorboard_only(k, v.metric.result())},
                                  **evaluator.result()}
            self._on_lav_end(params, all_metric_results)
            for cb in callbacks:
                cb.on_lav_end(params, all_metric_results)

            if not self._params.silent:
                # remove metrics which are for tensorboard use only
                metrics_to_print = {k: v for k, v in all_metric_results.items() if
                                    not self._model.tensorboard_handler.is_tensorboard_only(k, v)}
                print(json.dumps(metrics_to_print, indent=2))

            self.benchmark_results = predictor.benchmark_results
            yield all_metric_results

    def _on_sample_end(self, data_generator_params: DataGeneratorParams, sample: Sample):
        pass

    def _on_lav_end(self, data_generator_params: DataGeneratorParams, result):
        pass

    def extract_dump_data(self, sample: Sample):
        return sample.targets, sample.outputs
