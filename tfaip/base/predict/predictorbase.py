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
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Type, Iterable, Union, Optional, Any

import prettytable
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.utils import tf_utils

from tfaip.base.data.databaseparams import DataGeneratorParams
from tfaip.base.data.pipeline.datapipeline import DataPipeline, DataGenerator
from tfaip.base.data.pipeline.definitions import PipelineMode, Sample
from tfaip.base.device_config import DeviceConfig, distribute_strategy
from tfaip.base.predict.params import PredictorParams
from tfaip.util.multiprocessing.parallelmap import tqdm_wrapper
from tfaip.util.profiling import MeasureTime


if TYPE_CHECKING:
    from tfaip.base.data.data import DataBase

logger = logging.getLogger(__name__)


@dataclass
class PredictorBenchmarkResults:
    n_batches: float = 0
    n_samples: float = 0
    total_time: float = 1e-10
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


def wrap_post_processing(predict_fn):
    def wrapper(*args, **kwargs):
        predictor = args[0]
        return predictor.wrap_post_proc(predict_fn, *args, **kwargs)

    return wrapper


class PredictorBase(ABC):
    @classmethod
    def get_params_cls(cls) -> Type[PredictorParams]:
        return PredictorParams

    def __init__(self, params: PredictorParams, data: 'DataBase'):
        self._params = params
        self.device_config = DeviceConfig(self._params.device_params)
        self._data = data
        self.benchmark_results = PredictorBenchmarkResults()
        self._keras_model: Optional[keras.Model] = None

    @property
    def params(self):
        return self._params

    @property
    def data(self):
        return self._data

    def _load_model(self, model: Union[str, keras.Model], convert_to_input_output=True):
        if isinstance(model, str):
            model = keras.models.load_model(model, compile=False)

        # wrap to output inputs as outputs
        if convert_to_input_output:
            inputs = self._data.create_input_layers()
            outputs = model(inputs)
            if self._params.include_targets:
                targets = self._data.create_target_as_input_layers()
                joined = {**inputs, **targets}
                model = keras.models.Model(inputs=joined, outputs=(inputs, targets, outputs))
            else:
                model = keras.models.Model(inputs=inputs, outputs=(inputs, outputs))

        model.run_eagerly = self._params.run_eagerly
        return model

    def predict(self, params: DataGeneratorParams) -> Iterable[Sample]:
        return self.predict_pipeline(self._data.get_predict_data(params))

    @abstractmethod
    def _unwrap_batch(self, inputs, targets, outputs) -> Iterable[Sample]:
        raise NotImplementedError

    def predict_raw(self, inputs: Iterable[Any], *, size=None, batch_size=1) ->Iterable[Sample]:
        if size is None:
            try:
                size = len(inputs)
            except TypeError:
                logger.warning("Size not specified. Converting inputs to list to obtain size. Or implement size on inputs")
                inputs = len(inputs)
            size = len(inputs)

        class RawGenerator(DataGenerator):
            def __len__(self):
                return size

            def generate(self) -> Iterable[Sample]:
                return map(lambda x: Sample(inputs=x, meta={}), inputs)

        class RawInputsPipeline(DataPipeline):
            def create_data_generator(self) -> DataGenerator:
                return RawGenerator(mode=self.mode, params=self.generator_params)

        pipeline = RawInputsPipeline(
            PipelineMode.Prediction,
            self._data,
            DataGeneratorParams(batch_size=batch_size)
        )
        return self.predict_pipeline(pipeline)

    def predict_pipeline(self, pipeline: DataPipeline) -> Iterable[Sample]:
        with pipeline as rd:
            for r in tqdm_wrapper(
                    rd.process_output(self.predict_database(rd.input_dataset())),
                    progress_bar=self._params.progress_bar,
                    desc="Prediction",
                    total=len(rd),
            ):
                yield r

    @distribute_strategy
    def predict_database(self, dataset: tf.data.Dataset) -> Iterable[Sample]:
        if self._params.include_targets:
            dataset = dataset.map(lambda i, t: {**i, **t})

        if self._keras_model is None:
            raise ValueError("No model set. Call predictor.set_model(model)")
        with MeasureTime() as total_time:
            with self._keras_model.distribute_strategy.scope():
                data_handler = data_adapter.DataHandler(dataset)
                predict_function = self._keras_model.make_predict_function()
                for _, iterator in data_handler.enumerate_epochs():  # Single epoch.
                    with data_handler.catch_stop_iteration():
                        for step in data_handler.steps():
                            with MeasureTime() as batch_time:
                                r = predict_function(iterator)  # hack to access inputs
                                if self._params.include_targets:
                                    inputs, targets, r = tf_utils.to_numpy_or_python_type(r)
                                else:
                                    inputs, r = tf_utils.to_numpy_or_python_type(r)
                                    targets = {}  # No targets in normal prediction

                                batch_size = next(iter(inputs.values())).shape[0]
                                for sample in self._unwrap_batch(inputs, targets, r):
                                    self._on_sample_end(sample)
                                    if not self._params.silent:
                                        self._print_prediction(sample, logger.info)

                                    yield sample

                            self.benchmark_results.n_batches += 1
                            self.benchmark_results.n_samples += batch_size
                            self.benchmark_results.avg_time_per_batch += batch_time.duration
                            self.benchmark_results.avg_time_per_sample += batch_time.duration
                            self._on_step_end(Sample(inputs=inputs, outputs=r, targets=targets))

        self.benchmark_results.total_time = total_time.duration
        self.benchmark_results.avg_time_per_batch /= self.benchmark_results.n_batches
        self.benchmark_results.avg_time_per_sample /= self.benchmark_results.n_samples
        self.benchmark_results.batches_per_second = 1 / self.benchmark_results.avg_time_per_batch
        self.benchmark_results.samples_per_second = 1 / self.benchmark_results.avg_time_per_sample

        # print the output
        self._on_predict_end()

    @abstractmethod
    def _print_prediction(self, sample: Sample, print_fn):
        raise NotImplementedError

    def _on_sample_end(self, outputs):
        pass

    def _on_step_end(self, outputs):
        pass

    def _on_predict_end(self):
        pass
