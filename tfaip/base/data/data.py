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
from typing import Type, Dict, Iterable, Optional
import tensorflow.keras as keras
import tensorflow as tf
import logging

from tfaip.util.typing import AnyNumpy
from typeguard import typechecked

from tfaip.base.data.databaseparams import DataBaseParams, DataGeneratorParams
from tfaip.base.data.pipeline.datapipeline import DataPipeline
from tfaip.base.data.pipeline.dataprocessor import DataProcessorFactory
from tfaip.base.data.pipeline.definitions import PipelineMode
from tfaip.base.resource.manager import ResourceManager
from tfaip.base.resource.resource import Resource

logger = logging.getLogger(__name__)


def dict_to_input_layers(d: Dict[str, tf.TensorSpec]) -> Dict[str, keras.layers.Input]:
    for s, spec in d.items():
        assert isinstance(spec, tf.TensorSpec)

    return {k: keras.layers.Input(shape=v.shape, dtype=v.dtype, name=k) for k, v in d.items()}


def validate_specs(func):
    def wrapper(*args, **kwargs):
        retval = func(*args, **kwargs)
        for s, spec in retval.items():
            if len(spec.shape) == 0:
                # Reason: keras.predict automatically calls "expand_2d" which will expand 1d tensors (batch size) to
                # 2D tensors, which is not performed during training. This can lead to non-desired behaviour.
                raise ValueError(f"Shape of tensor spec must be at least one dimensional (excluding the "
                                 f"batch dimension), but got {spec.shape} for tensor {s}. Use (1, ) or [1] to "
                                 f"denote one dimensional data.")

        return retval
    return wrapper


class DataBase(ABC):
    """
    DataBase class to provide training and validation data.

    Override _input_layer_specs, and _output_layer_specs in a custom implementation
    """
    @staticmethod
    def get_params_cls() -> Type[DataBaseParams]:
        return DataBaseParams

    @classmethod
    def prediction_generator_params_cls(cls) -> Type[DataGeneratorParams]:
        return DataGeneratorParams

    @classmethod
    def get_default_params(cls) -> DataBaseParams:
        return cls.get_params_cls()()

    @classmethod
    @abstractmethod
    def data_processor_factory(cls) -> DataProcessorFactory:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def data_pipeline_cls(cls) -> Type[DataPipeline]:
        raise NotImplementedError

    def __init__(self, params: DataBaseParams):
        # Flags to change from sub-class
        self._auto_batch = True

        self._params = params
        self.resources = ResourceManager(params.resource_base_path_)
        self.resources.register_all(params)

        self._pipelines: Dict[PipelineMode, DataPipeline] = {}

    def preload(self, progress_bar=True):
        # Convert all pipelines to raw pipelines
        self._pipelines = {k: v.as_preloaded(progress_bar) for k, v in self._pipelines.items()}

    def print_params(self):
        logger.info("data_params=" + self._params.to_json(indent=2))

    def params(self) -> DataBaseParams:
        return self._params

    @typechecked
    def padding_values(self) -> Dict[str, AnyNumpy]:
        return self._padding_values()

    def _padding_values(self) -> Dict[str, AnyNumpy]:
        return {}

    @typechecked
    def get_train_data(self) -> DataPipeline:
        return self.get_pipeline(PipelineMode.Training, self._params.train)

    @typechecked
    def get_val_data(self) -> DataPipeline:
        return self.get_pipeline(PipelineMode.Evaluation, self._params.val)

    @typechecked
    def get_predict_data(self, params: Optional[DataGeneratorParams] = None) -> DataPipeline:
        return self.get_pipeline(PipelineMode.Prediction, params)

    @typechecked
    def get_targets_data(self, params: DataGeneratorParams) -> DataPipeline:
        return self.get_pipeline(PipelineMode.Targets, params)

    @typechecked
    def get_lav_datasets(self) -> Iterable[DataPipeline]:
        return self._list_lav_dataset()

    def create_pipeline(self, mode: PipelineMode, params: DataGeneratorParams) -> DataPipeline:
        return self.data_pipeline_cls()(mode,
                                        self,
                                        generator_params=params,
                                        )

    def get_pipeline(self, mode: PipelineMode, params: Optional[DataGeneratorParams] = None) -> DataPipeline:
        if mode in self._pipelines:
            return self._pipelines[mode]
        if params is None:
            raise ValueError("Pipe not yet instantiated")
        self._pipelines[mode] = self.create_pipeline(mode, params)
        return self._pipelines[mode]

    def _list_lav_dataset(self) -> Iterable[DataPipeline]:
        return [self.get_val_data()]

    @typechecked
    def create_input_layers(self) -> Dict[str, keras.layers.Input]:
        return dict_to_input_layers(self.input_layer_specs())

    @typechecked
    def create_target_as_input_layers(self) -> Dict[str, keras.layers.Input]:
        return dict_to_input_layers(self.target_layer_specs())

    @validate_specs
    @typechecked
    def input_layer_specs(self) -> Dict[str, tf.TensorSpec]:
        """
        The names of the inputs are exported, thus are the names if called from java

        :return: Dictionary of all inputs (omit the batch size), shape and dtype
        """
        return self._input_layer_specs()

    @validate_specs
    @typechecked
    def target_layer_specs(self) -> Dict[str, tf.TensorSpec]:
        """
        :return: Dictionary of all targets (omit the batch size), shape and dtype
        """
        return self._target_layer_specs()

    @abstractmethod
    def _input_layer_specs(self) -> Dict[str, tf.TensorSpec]:
        raise NotImplementedError

    @abstractmethod
    def _target_layer_specs(self) -> Dict[str, tf.TensorSpec]:
        raise NotImplementedError

    def register_resource_from_parameter(self, param_name: str) -> Resource:
        return self.resources.register(param_name, Resource(getattr(self._params, param_name)))

    def dump_resources(self, root_path: str, data_params_dict: dict):
        # dump resources and adjust the paths in the dumped dict
        data_params_dict['resource_base_path_'] = '.'
        self.resources.dump(root_path)
        for r_id, resource in self.resources.items():
            if r_id in data_params_dict:
                data_params_dict[r_id] = resource.dump_path
