from abc import ABC, abstractmethod
from typing import Type, Dict, Generator, Optional
import tensorflow.keras as keras
import tensorflow as tf
import logging
import os

from typeguard import typechecked

from tfaip.base.data.data_base_params import DataBaseParams

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
        params.validate()
        self._params = params
        self._current_val_list = 0
        self.resources_dir = os.getcwd()

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

    @staticmethod
    def _wrap_prefetch(dataset, size):
        if size > 0:
            return dataset.prefetch(size)
        return dataset

    @typechecked
    def get_train_data(self) -> tf.data.Dataset:
        if not self._is_entered:
            raise ValueError("get_train_data must be called within a 'with data:' statement")
        if not self._params.train_lists:
            raise ValueError("Empty train list in data.")

        return DataBase._wrap_prefetch(self._get_train_data(), self._params.train_prefetch).\
            take(compute_limit(self._params.train_limit, self._params.train_batch_size))

    @typechecked
    def get_val_data(self, val_list: Optional[str] = None) -> tf.data.Dataset:
        if val_list is None:
            val_list = self._params.val_list

        if not self._is_entered:
            raise ValueError("get_val_data must be called withing a 'with data:' statement")
        if not self._params.val_list:
            raise ValueError("Empty validation list in data.")

        return DataBase._wrap_prefetch(self._get_val_data(val_list), self._params.val_prefetch).\
            take(compute_limit(self._params.val_limit, self._params.val_batch_size))

    @typechecked
    def get_lav_datasets(self) -> Generator[tf.data.Dataset, None, None]:
        lav_lists = self._params.lav_lists if self._params.lav_lists else [self._params.val_list]
        for lav_list in lav_lists:
            yield self.get_val_data(lav_list)

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

    @typechecked
    def resource(self, path: str) -> str:
        return os.path.join(self.resources_dir, path)

    @abstractmethod
    def _get_train_data(self):
        raise NotImplemented

    @abstractmethod
    def _get_val_data(self, val_list: str):
        raise NotImplemented

    @abstractmethod
    def _input_layer_specs(self):
        raise NotImplemented

    @abstractmethod
    def _target_layer_specs(self):
        raise NotImplemented

    def dump_resources(self, root_path: str, resources_dir: str, data_params_dict: dict):
        """
        Override this method if your data requres resources for exporting the model (e.g. preproc in ATR)

        Override any path in data_params dict to be relative to root_path (e.g. resources_dir/resource).
        Then copy the resource into the root path.

        :param root_path: the root path (all paths should be stored relative to this dir)
        :param resources_dir: the subdir there to store the resource
        :param data_params_dict: serialized params to export (change all absolute paths of exported resources relative to root_path)
        """
        pass

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
