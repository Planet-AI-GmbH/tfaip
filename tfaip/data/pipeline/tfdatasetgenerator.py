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
"""Implementation of TFDatasetGenerator"""
import json
from functools import partial
from typing import Callable, Iterable, TYPE_CHECKING, TypeVar, Generic, Union, List

import numpy as np
from tfaip import PipelineMode, Sample
from tfaip.util.json_helper import TFAIPJsonEncoder

if TYPE_CHECKING:
    import tensorflow as tf  # pylint: disable=ungrouped-imports
    from tfaip.data.pipeline.datapipeline import DataPipeline


TDataPipeline = TypeVar("TDataPipeline", bound="DataPipeline")


class TFDatasetGenerator(Generic[TDataPipeline]):
    """
    Purpose: Transformation of a pure-python pipeline into the tf.data.Dataset world

    Usage: Used in the RunningDataPipeline to instantiate tf.data.Dataset

    Customization: Customize this class (in DataPipeline) if you want to apply tf.data.Dataset.map calls to transform
                   the data before batching/prefetching/...
    """

    def __init__(self, data_pipeline: TDataPipeline):
        self.data_pipeline = data_pipeline
        self.mode = data_pipeline.mode

    @property
    def data_params(self):
        return self.data_pipeline.data_params

    def _transform(self, dataset: "tf.data.Dataset") -> "tf.data.Dataset":
        # Override this, when you want to apply additional transformations on the dataset
        # Usually calling dataset.map(...)
        # Note: you should handle different cases for the different PipelineModels which will generate a different
        # dataset shape
        return dataset

    def input_layer_specs(self):
        # Override this, when the generator yields other shapes/types than the final data pipeline (input to the model)
        return self.data_pipeline.data.input_layer_specs()

    def target_layer_specs(self):
        # Override this, when the generator yields other shapes/types than the final data pipeline (input to the model)
        return self.data_pipeline.data.target_layer_specs()

    def meta_layer_specs(self):
        return self.data_pipeline.data.meta_layer_specs()

    def create(self, generator_fn: Callable[[], Iterable[Sample]], yields_batches=False) -> "tf.data.Dataset":
        """

        Args:
            generator_fn: Callable that creates the actual generator
            yields_batches: True if the samples comprise already batched data

        Returns:
            The tf.data.Dataset
        """
        # Local input so that not imported in spawned processes
        import tensorflow as tf  # pylint: disable=import-outside-toplevel

        def wrap(sample):
            if isinstance(sample, Sample):
                if yields_batches:
                    meta = sample.meta
                else:
                    meta = {"meta": np.asarray([json.dumps(sample.meta, cls=TFAIPJsonEncoder)])}
                if self.mode == PipelineMode.PREDICTION:
                    return sample.inputs, meta
                elif self.mode == PipelineMode.TARGETS:
                    return sample.targets, meta
                else:
                    return sample.inputs, sample.targets, meta
            else:
                return sample

        if self.mode == PipelineMode.PREDICTION:
            output_signature = (self.input_layer_specs(), self.meta_layer_specs())
        elif self.mode == PipelineMode.TARGETS:
            output_signature = (self.target_layer_specs(), self.meta_layer_specs())
        else:
            output_signature = (self.input_layer_specs(), self.target_layer_specs(), self.meta_layer_specs())

        if yields_batches:
            output_signature = tf.nest.map_structure(
                lambda x: tf.TensorSpec(shape=(None,) + x.shape, dtype=x.dtype, name=x.name), output_signature
            )

        # create the tf.data.Dataset and apply optional additional transformations (overwritten by implementations)
        def flatten(x):
            return tuple(tf.nest.flatten(x))

        def unflatten(*args):
            return tf.nest.pack_sequence_as(output_signature, args)

        dataset = tf.data.Dataset.from_generator(
            lambda: map(flatten, map(wrap, generator_fn())),
            output_signature=flatten(output_signature),
        )
        dataset = dataset.map(unflatten)
        dataset = self._transform(dataset)
        return dataset
