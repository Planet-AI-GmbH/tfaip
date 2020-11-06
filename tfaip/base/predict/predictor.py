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
import os
from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Type, Dict, List, Generator, Union

import prettytable
from dataclasses_json import dataclass_json
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.utils import tf_utils

from tfaip.base.device_config import DeviceConfig, DeviceConfigParams, distribute_strategy
from tfaip.util.time import MeasureTime
from tfaip.util.typing import AnyNumpy

if TYPE_CHECKING:
    from tfaip.base.data.data import DataBase
    from tfaip.base.model import ModelBase

logger = logging.getLogger(__name__)


@dataclass_json
@dataclass
class PredictorParams:
    max_iter: int = -1
    model_path_: str = None

    device_params: DeviceConfigParams = field(default_factory=lambda: DeviceConfigParams())

    silent: bool = False
    run_eagerly: bool = False


@dataclass
class PredictorBenchmarkResults:
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


class Predictor(ABC):
    @classmethod
    def get_params_cls(cls) -> Type[PredictorParams]:
        return PredictorParams

    def __init__(self, params: PredictorParams, model: 'ModelBase', data: 'DataBase'):
        assert params.model_path_
        self._params = params
        self.device_config = DeviceConfig(self._params.device_params)
        self._model = model
        self._data = data
        self._keras_model: keras.Model = keras.models.load_model(os.path.join(self._params.model_path_, 'serve'),
                                                                 compile=False,
                                                                 custom_objects=model.get_all_custom_objects())
        self._keras_model.run_eagerly = params.run_eagerly
        self.benchmark_results = PredictorBenchmarkResults()

    def predict_list(self, predict_list: str) -> Generator[Dict[str, AnyNumpy], None, None]:
        return self.predict_lists([predict_list])

    def predict_lists(self, predict_lists: List[str]) -> Generator[Dict[str, AnyNumpy], None, None]:
        with self._data:
            for p_list in predict_lists:
                for r in self.predict_database(self._data.get_predict_data(p_list).take(self._params.max_iter)):
                    yield r

    @distribute_strategy
    def predict_database(self, dataset: Union[tf.data.Dataset, Generator[Dict[str, AnyNumpy], None, None]]) -> Generator[Dict[str, AnyNumpy], None, None]:
        with MeasureTime() as total_time:
            with self._keras_model.distribute_strategy.scope():
                data_handler = data_adapter.DataHandler(dataset)
                predict_function = self._keras_model.make_predict_function()
                for _, iterator in data_handler.enumerate_epochs():  # Single epoch.
                    with data_handler.catch_stop_iteration():
                        with MeasureTime() as batch_time:
                            for step in data_handler.steps():
                                r = predict_function(iterator)
                                r = tf_utils.to_numpy_or_python_type(r)
                                batch_size = self._data.params().val_batch_size
                                self.benchmark_results.n_batches += 1
                                self.benchmark_results.n_samples += batch_size
                                self.benchmark_results.avg_time_per_batch += batch_time.duration
                                self.benchmark_results.avg_time_per_sample += batch_time.duration
                                for i in range(batch_size):
                                    un_batched_outputs = {k: v[i] for k, v in r.items()}
                                    self._on_sample_end(un_batched_outputs)
                                    if not self._params.silent:
                                        self._model.print_prediction(un_batched_outputs, self._data)

                                    yield un_batched_outputs

                                self._on_step_end(r)

        self.benchmark_results.total_time = total_time.duration
        self.benchmark_results.avg_time_per_batch /= self.benchmark_results.n_batches
        self.benchmark_results.avg_time_per_sample /= self.benchmark_results.n_samples
        self.benchmark_results.batches_per_second = 1 / self.benchmark_results.avg_time_per_batch
        self.benchmark_results.samples_per_second = 1 / self.benchmark_results.avg_time_per_sample

        # print the output
        self._on_predict_end()

    def _on_sample_end(self, outputs):
        pass

    def _on_step_end(self, outputs):
        pass

    def _on_predict_end(self):
        pass
