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
"""Module that defines DataBase
"""
import logging
from abc import ABC, abstractmethod
from typing import Type, Dict, Optional, TypeVar, Generic, Callable

import tensorflow as tf
import tensorflow.keras as keras
from typeguard import typechecked

from tfaip.data.databaseparams import DataBaseParams, DataGeneratorParams, DataPipelineParams
from tfaip import PipelineMode
from tfaip.data.pipeline.datapipeline import DataPipeline
from tfaip.resource.manager import ResourceManager
from tfaip.resource.resource import Resource
from tfaip.util.generic_meta import CollectGenericTypes
from tfaip.util.tfaipargparse import post_init
from tfaip.util.tftyping import AnyTensor
from tfaip.util.typing import AnyNumpy

logger = logging.getLogger(__name__)


def dict_to_input_layers(d: Dict[str, tf.TensorSpec]) -> Dict[str, keras.layers.Input]:
    for _, spec in d.items():
        assert isinstance(spec, tf.TensorSpec)

    return {k: keras.layers.Input(shape=v.shape, dtype=v.dtype, name=k) for k, v in d.items()}


def validate_specs(func):
    def wrapper(*args, **kwargs):
        retval = func(*args, **kwargs)
        for s, spec in retval.items():
            if len(spec.shape) == 0:
                # Reason: keras.predict automatically calls "expand_2d" which will expand 1d tensors (batch size) to
                # 2D tensors, which is not performed during training. This can lead to non-desired behaviour.
                raise ValueError(
                    f"Shape of tensor spec must be at least one dimensional (excluding the "
                    f"batch dimension), but got {spec.shape} for tensor {s}. Use (1, ) or [1] to "
                    f"denote one dimensional data."
                )

        return retval

    return wrapper


TDP = TypeVar("TDP", bound=DataBaseParams)


class DataBase(Generic[TDP], ABC, metaclass=CollectGenericTypes):
    """
    DataBase class to provide training and validation data.

    Override _input_layer_specs, and _output_layer_specs in a custom implementation
    """

    @classmethod
    def params_cls(cls) -> Type[TDP]:
        return cls.__generic_types__[TDP.__name__]

    @classmethod
    def default_params(cls) -> TDP:
        return cls.params_cls()()

    @classmethod
    def data_pipeline_cls(cls) -> Type[DataPipeline]:
        return DataPipeline

    def __init__(self, params: TDP):
        post_init(params)
        self._params = params
        self.resources = ResourceManager(params.resource_base_path)
        self.resources.register_all(params)
        post_init(params)

        self._pipelines: Dict[PipelineMode, DataPipeline] = {}

    def preload(self, progress_bar=True):
        # Convert all pipelines to raw pipelines
        self._pipelines = {k: v.as_preloaded(progress_bar) for k, v in self._pipelines.items()}

    def print_params(self):
        logger.info("data={}", self._params.to_json(indent=2))

    @property
    def params(self) -> TDP:
        return self._params

    @typechecked
    def padding_values(self) -> Dict[str, AnyNumpy]:
        return self._padding_values()

    def _padding_values(self) -> Dict[str, AnyNumpy]:
        return {}

    def element_length_fn(self) -> Callable[[Dict[str, AnyTensor]], AnyTensor]:
        """Element length for bucked_by_sequence_length"""
        raise Exception("Implement this function if you want to use buckets")

    def create_pipeline(self, pipeline_params: DataPipelineParams, params: DataGeneratorParams) -> DataPipeline:
        return self.data_pipeline_cls()(
            pipeline_params,
            self,
            generator_params=params,
        )

    def get_or_create_pipeline(
        self, pipeline_params: DataPipelineParams, params: Optional[DataGeneratorParams]
    ) -> DataPipeline:
        mode = pipeline_params.mode
        if mode in self._pipelines:
            return self._pipelines[mode]
        if params is None:
            self._pipelines[mode] = None  # No data in this pipeline
        else:
            self._pipelines[mode] = self.create_pipeline(pipeline_params, params)
        return self._pipelines[mode]

    def pipeline_by_mode(self, mode: PipelineMode) -> DataPipeline:
        return self._pipelines[mode]

    @typechecked
    def create_input_layers(self) -> Dict[str, keras.layers.Input]:
        return dict_to_input_layers(self.input_layer_specs())

    @typechecked
    def create_target_as_input_layers(self) -> Dict[str, keras.layers.Input]:
        return dict_to_input_layers(self.target_layer_specs())

    @typechecked
    def create_meta_as_input_layers(self) -> Dict[str, keras.layers.Input]:
        return dict_to_input_layers(self.meta_layer_specs())

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

    @validate_specs
    @typechecked
    def meta_layer_specs(self) -> Dict[str, tf.TensorSpec]:
        return self._meta_layer_specs()

    @abstractmethod
    def _input_layer_specs(self) -> Dict[str, tf.TensorSpec]:
        raise NotImplementedError

    @abstractmethod
    def _target_layer_specs(self) -> Dict[str, tf.TensorSpec]:
        raise NotImplementedError

    def _meta_layer_specs(self) -> Dict[str, tf.TensorSpec]:
        return {"meta": tf.TensorSpec(shape=[1], dtype=tf.string)}

    def register_resource_from_parameter(self, param_name: str) -> Resource:
        return self.resources.register(param_name, Resource(getattr(self._params, param_name)))

    def dump_resources(self, root_path: str, data_params_dict: dict):
        # dump resources and adjust the paths in the dumped dict
        data_params_dict["resource_base_path"] = "."
        self.resources.dump(root_path)
        for r_id, resource in self.resources.items():
            if r_id in data_params_dict:
                data_params_dict[r_id] = resource.dump_path
