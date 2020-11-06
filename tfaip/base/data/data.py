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
from abc import ABC, abstractmethod
from typing import Type, Dict, Generator, Optional
import tensorflow.keras as keras
import tensorflow as tf
import logging
import os

from typeguard import typechecked

from tfaip.base.data.data_base_params import DataBaseParams
from tfaip.base.resource.manager import ResourceManager
from tfaip.base.resource.resource import Resource

logger = logging.getLogger(__name__)


def dict_to_input_layers(d: Dict[str, tf.TensorSpec]) -> Dict[str, keras.layers.Input]:
    for s, spec in d.items():
        assert isinstance(spec, tf.TensorSpec)

    return {k: keras.layers.Input(shape=v.shape, dtype=v.dtype, name=k) for k, v in d.items()}


def compute_limit(limit, batch_size):
    assert(limit != 0)
    if limit < 0:
        return limit  # no limit
    else:
        return -(-limit // batch_size)  # ceiled integer div => 1 // 3 = 1; 3 // 3 => 1; 4 // 3 = 2


class DataBase(ABC):
    """
    DataBase class to provide training and validation data.

    Override _get_train_data, _get_val_data, _input_layer_specs, and _output_layer_specs in a custom implementation
    """
    @staticmethod
    def get_params_cls() -> Type[DataBaseParams]:
        return DataBaseParams

    def __init__(self, params: DataBaseParams):
        # Flags to change from sub-class
        self._auto_batch = True

        params.validate()
        self._params = params
        self._current_val_list = 0
        self.resources = ResourceManager(params.resource_base_path_)

        self._is_entered = False
        self._pipelines = []

    def __enter__(self):
        if self._is_entered:
            raise ValueError("Calling with DataBase was called stacked.")

        self._is_entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._is_entered = False
        for pipeline in self._pipelines:
            pipeline.join()
        self._pipelines = []

    def register_pipeline(self, pipeline):
        self._pipelines.append(pipeline)

    def print_params(self):
        logger.info("data_params=" + self._params.to_json(indent=2))

    def params(self) -> DataBaseParams:
        return self._params

    def _padding_values(self) -> Dict[str, float]:
        return {}

    def _wrap_padded_batch(self, dataset: tf.data.Dataset, batch_size: int, drop_remainder: bool, predict: bool):
        pad_values = self._padding_values()

        def default(dtype):
            return '' if dtype == tf.string else 0

        if predict:
            shapes = {k: v.shape for k, v in self.input_layer_specs().items()}
            values = {k: tf.constant(pad_values.get(k, default(v.dtype)), dtype=v.dtype) for k, v in self.input_layer_specs().items()}
        else:
            shapes = (
                {k: v.shape for k, v in self.input_layer_specs().items()},
                {k: v.shape for k, v in self.target_layer_specs().items()},
            )
            values = (
                         {k: tf.constant(pad_values.get(k, default(v.dtype)), dtype=v.dtype) for k, v in self.input_layer_specs().items()},
                         {k: tf.constant(pad_values.get(k, default(v.dtype)), dtype=v.dtype) for k, v in self.target_layer_specs().items()},
                     )

        return dataset.padded_batch(batch_size, shapes, values, drop_remainder=drop_remainder)

    def _wrap_dataset(self, dataset, batch_size, prefetch, limit, drop_remainder, predict=False):
        if self._auto_batch:
            dataset = self._wrap_padded_batch(dataset, batch_size, drop_remainder, predict=predict)
        if prefetch > 0:
            dataset = dataset.prefetch(prefetch)
        dataset = dataset.take(compute_limit(limit, batch_size))
        return dataset

    @typechecked
    def get_train_data(self) -> tf.data.Dataset:
        if not self._is_entered:
            raise ValueError("get_train_data must be called within a 'with data:' statement")
        if not self._params.train_lists:
            raise ValueError("Empty train list in data.")
        return self._wrap_dataset(self._get_train_data(),
                                  batch_size=self._params.train_batch_size,
                                  prefetch=self._params.train_prefetch,
                                  limit=self._params.train_limit,
                                  drop_remainder=self._params.train_batch_drop_remainder,
                                  )

    @typechecked
    def get_val_data(self, val_list: Optional[str] = None) -> tf.data.Dataset:
        if val_list is None:
            val_list = self._params.val_list

        if not self._is_entered:
            raise ValueError("get_val_data must be called withing a 'with data:' statement")
        if not self._params.val_list:
            raise ValueError("Empty validation list in data.")

        return self._wrap_dataset(self._get_val_data(val_list),
                                  batch_size=self._params.val_batch_size,
                                  prefetch=self._params.val_prefetch,
                                  limit=self._params.val_limit,
                                  drop_remainder=self._params.val_batch_drop_remainder,
                                  )

    @typechecked
    def get_lav_datasets(self) -> Generator[tf.data.Dataset, None, None]:
        lav_lists = self._params.lav_lists if self._params.lav_lists else [self._params.val_list]
        for lav_list in lav_lists:
            yield self.get_val_data(lav_list)

    @typechecked
    def get_predict_data(self, predict_list: Optional[str] = None) -> tf.data.Dataset:
        if predict_list is None:
            predict_list = self._params.lav_lists[0] if self._params.lav_lists else self._params.val_list

        if not self._is_entered:
            raise ValueError("get_val_data must be called withing a 'with data:' statement")
        if not predict_list:
            raise ValueError("Empty prediction list in")

        return self._wrap_dataset(self._get_predict_data(predict_list),
                                  batch_size=self._params.val_batch_size,
                                  prefetch=self._params.val_prefetch,
                                  limit=self._params.val_limit,
                                  drop_remainder=False,
                                  predict=True,
                                  )

    @typechecked
    def create_input_layers(self) -> Dict[str, keras.layers.Input]:
        return dict_to_input_layers(self.input_layer_specs())

    @typechecked
    def create_target_as_input_layers(self) -> Dict[str, keras.layers.Input]:
        return dict_to_input_layers(self.target_layer_specs())

    @typechecked
    def input_layer_specs(self) -> Dict[str, tf.TensorSpec]:
        """
        The names of the inputs are exported, thus are the names if called from java

        :return: Dictionary of all inputs (omit the batch size), shape and dtype
        """
        return self._input_layer_specs()

    @typechecked
    def target_layer_specs(self) -> Dict[str, tf.TensorSpec]:
        """
        :return: Dictionary of all targets (omit the batch size), shape and dtype
        """
        return self._target_layer_specs()

    @abstractmethod
    def _get_train_data(self):
        raise NotImplemented

    @abstractmethod
    def _get_val_data(self, val_list: str):
        raise NotImplemented

    def _get_predict_data(self, predict_list: str):
        return self._get_val_data(predict_list).map(lambda inputs, targets: inputs)

    @abstractmethod
    def _input_layer_specs(self):
        raise NotImplemented

    @abstractmethod
    def _target_layer_specs(self):
        raise NotImplemented

    def register_resource_from_parameter(self, param_name: str) -> Resource:
        return self.resources.register(Resource(param_name, getattr(self._params, param_name)))

    def dump_resources(self, root_path: str, data_params_dict: dict):
        # dump resources and adjust the paths in the dumped dict
        data_params_dict['resource_base_path_'] = '.'
        self.resources.dump(root_path)
        for r_id, resource in self.resources.items():
            if r_id in data_params_dict:
                data_params_dict[r_id] = resource.dump_path

    def set_val_list(self, idx: int):
        self._params.val_list = self._params.lav_lists[idx]

    def reset_val_list(self):
        if self._params.lav_lists:
            self._params.val_list = self._params.lav_lists[0]

    def next_val_list(self):
        if not self._params.lav_lists or self._current_val_list + 1 >= len(self._params.lav_lists):
            return False
        self._current_val_list += 1
        self._params.val_list = self._params.lav_lists[0]
        return True


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    class TestData(DataBase):
        def _get_train_data(self):
            pass

        def _get_val_data(self, val_list):
            pass

    data = TestData(DataBaseParams(
        val_list='test',
        lav_lists=['asdf'],
    ))
    data.print_params()
