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
import json
import logging
import os
from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Type, Dict, List, Callable, Generator

import prettytable
import tensorflow.keras as keras
from dataclasses_json import dataclass_json

from tfaip.base.device_config import DeviceConfig, DeviceConfigParams, distribute_strategy
from tfaip.base.lav.callbacks.lav_callback import LAVCallback
from tfaip.util.file.oshelper import ChDir
from tfaip.util.time import MeasureTime

if TYPE_CHECKING:
    from tfaip.base import DataBase, ModelBase

logger = logging.getLogger(__name__)


@dataclass_json
@dataclass
class LAVParams:
    max_iter: int = -1

    model_path_: str = None

    device_params: DeviceConfigParams = field(default_factory=lambda: DeviceConfigParams())

    silent: bool = False


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

    def accumulate_dict_sum(self, new_values: Dict[str, List[float]], sample_weights: Dict[str, List[float]]):
        def weighted_sum(values, weights):
            if weights is None:
                return sum(values)

            return sum(a * w for a, w in zip(values, weights))

        if len(self.running_sum) == 0:
            self.running_sum = {k: 0 for k, v in new_values.items()}
            self.running_weight = {k: 0 for k, v in new_values.items()}

        self.running_sum = {k: (self.running_sum[k] + weighted_sum(v, sample_weights.get(k, None))) for k, v in new_values.items()}
        self.running_weight = {k: (v + (len(new_values[k]) if k not in sample_weights else sum(sample_weights[k]))) for k, v in
                               self.running_weight.items()}

    def final(self):
        return {k: v / self.running_weight[k] for k, v in self.running_sum.items()}


@dataclass
class LAVBenchmarkResults:
    n_batches: float = 0
    n_samples: float = 0
    total_time: float = 0
    avg_time_per_batch: float = 0
    avg_time_per_sample: float = 0
    batches_per_second: float = 0
    samples_per_second: float = 0

    def pretty_print(self):
        table = prettytable.PrettyTable(['', "Total", "Batch", "Sample"])
        table.add_row(['Count', 1, self.n_batches, self.n_samples])
        table.add_row(['Time Per', self.total_time, self.avg_time_per_batch, self.avg_time_per_sample])
        table.add_row(['Per Second', 1 / self.total_time, self.batches_per_second, self.samples_per_second])
        print(table)


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
    def get_params_cls(cls) -> Type[LAVParams]:
        return LAVParams

    def __init__(self, params: LAVParams, data_fn: Callable[[], 'DataBase'], model_fn: Callable[[], 'ModelBase']):
        assert params.model_path_
        self._params = params
        self._data_fn = data_fn
        self._model_fn = model_fn
        self.device_config = DeviceConfig(self._params.device_params)
        self._data: 'DataBase' = None
        self._model: 'ModelBase' = None
        self.benchmark_results = LAVBenchmarkResults()

    @distribute_strategy
    def run(self, model: keras.Model = None, silent=False, run_eagerly=False, callbacks: List[LAVCallback] = None) -> Generator[Dict[str, float], None, None]:
        callbacks = callbacks if callbacks else []
        with ChDir(os.path.join(self._params.model_path_)):
            # resources are located in parent dir
            self._data = self._data_fn()
        self._model = self._model_fn()
        for cb in callbacks:
            cb.lav, cb.data, cb.model = self, self._data, self._model

        if run_eagerly:
            logger.warning("Running in eager mode. Use this only for debugging, since the graph of the saved model "
                           "might get changed due to 'reconstruction' of the graph")
            custom_objects = self._model.__class__.get_all_custom_objects()
        else:
            custom_objects = None

        _keras_model: keras.Model = model if model else keras.models.load_model(os.path.join(self._params.model_path_, 'serve'), compile=False, custom_objects=custom_objects)
        _keras_model.run_eagerly = run_eagerly
        # create a new keras model that uses the inputs and outputs of the loaded model but adds the targets of the
        # dataset. Then create the metrics as output of the new model
        with self._data:
            eval_inputs = {**_keras_model.input, **self._data.create_target_as_input_layers()}
            metric_outputs = self._model.extended_metric(eval_inputs, _keras_model.output)
            simple_metrics = self._model.metric()
            eval_model = keras.Model(eval_inputs, {**metric_outputs, **_keras_model.output})
            eval_model.run_eagerly = run_eagerly

            # accumulate the mean
            metrics_accum = MetricsAccumulator()
            for val_data in self._data.get_lav_datasets():
                self.benchmark_results = LAVBenchmarkResults()
                val_data = val_data.take(self._params.max_iter)
                with MeasureTime() as total_time:
                    for step, (inputs, targets) in enumerate(val_data):
                        combined_batch = {**inputs, **targets}
                        with MeasureTime() as time_of_batch:
                            r = eval_model.predict_on_batch(combined_batch)
                        batch_size = next(iter(inputs.values())).shape[0]  # Take an arbitrary input to get the first dimension
                        self.benchmark_results.n_batches += 1
                        self.benchmark_results.n_samples += batch_size
                        self.benchmark_results.avg_time_per_batch += time_of_batch.duration
                        self.benchmark_results.avg_time_per_sample += time_of_batch.duration
                        for i in range(batch_size):
                            un_batched_inputs = {k: v[i].numpy() for k, v in inputs.items()}
                            un_batched_targets = {k: v[i].numpy() for k, v in targets.items()}
                            un_batched_outputs = {k: v[i] for k, v in r.items()}
                            self._on_sample_end(un_batched_inputs, un_batched_targets, un_batched_outputs)
                            for cb in callbacks:
                                cb.on_sample_end(un_batched_inputs, un_batched_targets, un_batched_outputs)
                            if not self._params.silent:
                                self._model.print_evaluate(un_batched_inputs, un_batched_outputs, un_batched_targets, self._data)

                        sample_weights = self._model.sample_weights(inputs, targets)
                        for k, metric in simple_metrics.items():
                            metric.metric.update_state(combined_batch[metric.target], r[metric.output], sample_weights.get(k, None))

                        metrics_r = {k: r[k] for k in metric_outputs.keys()}
                        sample_weights = {k: v.numpy() for k, v in sample_weights.items()}
                        metrics_accum.accumulate_dict_sum(metrics_r, sample_weights)
                        self._on_step_end(inputs, targets, r, metrics_r)
                        for cb in callbacks:
                            cb.on_step_end(inputs, targets, r, metrics_r)

                self.benchmark_results.total_time = total_time.duration
                self.benchmark_results.avg_time_per_batch /= self.benchmark_results.n_batches
                self.benchmark_results.avg_time_per_sample /= self.benchmark_results.n_samples
                self.benchmark_results.batches_per_second = 1 / self.benchmark_results.avg_time_per_batch
                self.benchmark_results.samples_per_second = 1 / self.benchmark_results.avg_time_per_sample

                # print the output
                all_metric_results = {**metrics_accum.final(), **{k: float(v.metric.result().numpy()) for k, v in simple_metrics.items()}}
                self._on_lav_end(all_metric_results)
                for cb in callbacks:
                    cb.on_lav_end(all_metric_results)
                if not self._params.silent:
                    print(json.dumps(all_metric_results, indent=2))
                yield all_metric_results

    def _on_sample_end(self, inputs, targets, outputs):
        pass

    def _on_step_end(self, inputs, targets, outputs, metrics):
        pass

    def _on_lav_end(self, result):
        pass
