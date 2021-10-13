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
"""Implementation various classes that represent a running pipeline as context.

The classes ensure that created processes get cleaned up by enforcing `with`-clauses.
"""
import itertools
import json
from types import GeneratorType
from typing import Iterable, List, Optional

import numpy as np
import tensorflow as tf
from tensorflow.python.data.experimental import bucket_by_sequence_length

from tfaip import Sample, PipelineMode
from tfaip.data.data import DataBase
from tfaip.data.databaseparams import DataPipelineParams
from tfaip.data.pipeline.datagenerator import DataGenerator
from tfaip.data.pipeline.processor.dataprocessor import is_valid_sample
from tfaip.data.pipeline.processor.params import DataProcessorPipelineParams
from tfaip.data.pipeline.tfdatasetgenerator import TFDatasetGenerator
from tfaip.util.json_helper import TFAIPJsonEncoder


def compute_limit(limit, batch_size):
    assert limit != 0, "The limit must not be 0 if calling this method."
    if limit < 0:
        return limit  # no limit
    else:
        return -(-limit // batch_size)  # ceiled integer div => 1 // 3 = 1; 3 // 3 => 1; 4 // 3 = 2


class OutputSamplesGenerator:
    """Apply the post-processing pipeline to the given samples.

    Note:
        This class should not be used as stand-alone, see `DataPipeline.process_output` instead.

    Usage:
        ```
        with OutputSamplesGenerator(in_samples, ...) as out_samples:
            for s in out_samples:
                print(s)
        ```
    """

    def __init__(
        self,
        samples: Iterable[Sample],
        processor_pipeline_params: DataProcessorPipelineParams,
        data: DataBase,
        pipeline_params: DataPipelineParams,
    ):
        self.samples = samples
        self.processor_pipeline_params = processor_pipeline_params
        self.pipeline_params = pipeline_params
        self.data = data

    def __enter__(self):
        self.generator = self._process_output()
        return self.generator

    def __exit__(self, exc_type, exc_val, exc_tb):
        if isinstance(self.generator, GeneratorType):
            self.generator.close()

    def _process_output(self) -> Iterable[Sample]:
        output_pipeline = self.processor_pipeline_params.create(self.pipeline_params, self.data.params)
        samples = self.samples

        def extract_meta(sample: Sample) -> Sample:
            meta = sample.meta or {}
            if "meta" in sample.inputs:
                input_meta = sample.inputs["meta"]
                if isinstance(input_meta, (list, np.ndarray)):
                    assert len(input_meta) == 1, "This must be one, just be sure"
                    input_meta = input_meta[0]

                meta.update(**json.loads(input_meta))
            return sample.new_meta(meta)

        samples = map(extract_meta, samples)

        if output_pipeline:
            return output_pipeline.apply(samples)
        return samples


class InputSamplesGenerator:
    """Obtain the samples after applying the pre-proc pipeline

    Note:
        This class should not be used as stand-alone. See `DataPipeline.generate_input_samples` instead.

    Usage:

        To obtain a `Generator[Sample]`
        ```
        input_samples_generator = InputSamplesGenerator(...)
        with input_samples_generator as samples:
            for sample in samples:
                print(sample)
        ```

        Alternatively to obtain a `tf.data.Dataset` yielding batched samples (input of the net)
        ```
        dataset = InputSamplesGenerator(...).as_dataset()
        for batch in dataset.as_numpy_iterator():
            print(batch)
        ```
    Params:
        auto_repeat: if None, auto repeat defaults to True on Training else to False.
    """

    def __init__(
        self,
        processor_pipeline_params: DataProcessorPipelineParams,
        data: DataBase,
        data_generator: DataGenerator,
        pipeline_params: DataPipelineParams,
        mode: PipelineMode,
        auto_repeat: bool = False,
    ):
        self.processor_pipeline_params = processor_pipeline_params
        self.data_params = data.params
        self.data = data
        self.pipeline_params = pipeline_params
        self.mode = mode
        self.data_generator = data_generator
        self.auto_repeat = auto_repeat
        self.running_pipeline = None
        self.running_samples_generator = None
        self._stack = None

    def as_dataset(self, tf_dataset_generator: TFDatasetGenerator) -> Optional[tf.data.Dataset]:
        """
        Obtain the batched, padded, prefetched tf.data.Dataset of the input.

        Returns:
            A tf.data.Dataset or None if the dataset is empty
        """
        if len(self) == 0:
            # Empty set
            return None

        def generator():
            with self as samples:
                for s in samples:
                    yield s

        dataset = tf_dataset_generator.create(generator, self.data_generator.yields_batches())
        dataset = _wrap_dataset(
            self.mode, dataset, self.pipeline_params, self.data, self.data_generator.yields_batches()
        )
        return dataset

    def __len__(self):
        """
        The number of samples produced by the data generator.
        Note that this might not correspond to the number of examples produced by the pre-proc-pipeline if
        `GeneratingDataProcessors` are used.
        """
        real_len = len(self.data_generator)
        if self.pipeline_params.limit > 0:
            return min(self.pipeline_params.limit, real_len)
        return real_len

    def __enter__(self):
        data_generator = self.data_generator
        auto_repeat = self.auto_repeat

        if auto_repeat is None:
            auto_repeat = self.mode == PipelineMode.TRAINING and self.pipeline_params.limit < 0

        self.running_pipeline = self.processor_pipeline_params.create(self.pipeline_params, self.data_params)
        self.running_samples_generator = self._generate_input_samples()
        return self.running_samples_generator

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close the iterator, so that it cannot be used anymore.
        # This will cleanup anything within (e.g. threads, processes, ...)
        self.running_samples_generator.close()

    def _generate_input_samples(self) -> Iterable[Sample]:

        # Do-while-loop required if auto_repeat==True
        while True:
            # get the Iterator[Samples] from the data generator and apply the limit if set
            generate = self.data_generator.generate()
            if self.pipeline_params.limit > 0:
                generate = itertools.islice(generate, self.pipeline_params.limit)

            # Apply the input pipeline
            if self.running_pipeline is not None:
                generate = self.running_pipeline.apply(generate)

            # Generate the samples (skip invalid samples)
            for s in generate:
                if self.data_generator.yields_batches():
                    assert isinstance(s, list), f"Batched mode. Invalid type. Expected List[Sample] but got {type(s)}"
                    # batched, filter invalid samples in batch, and yield the remainder, if something remains
                    s = list(filter(lambda x: is_valid_sample(x, self.mode), s))
                    if len(s) > 0:
                        yield self._pad_batched_samples(s)
                else:
                    assert isinstance(s, Sample), f"Sample mode. Invalid type. Expected Sample but got {type(s)}."
                    if is_valid_sample(s, self.mode):
                        yield s

            if not self.auto_repeat:
                # Stop of auto repeat is false
                break

    def _pad_batched_samples(self, samples: List[Sample]) -> Sample:
        """Batches and pads the content of samples"""
        data = self.data

        def pack_meta(meta):
            return {"meta": np.asarray([json.dumps(meta, cls=TFAIPJsonEncoder)])}

        if self.mode == PipelineMode.PREDICTION:
            output_signature = (data.dataset_input_layer_specs(), data.dataset_meta_layer_specs())
            extract = lambda s: (s.inputs, pack_meta(s.meta))
            to_sample = lambda i, m: Sample(inputs=i, meta=m)
        elif self.mode == PipelineMode.TARGETS:
            output_signature = (data.dataset_target_layer_specs(), data.dataset_meta_layer_specs())
            extract = lambda s: (s.targets, pack_meta(s.meta))
            to_sample = lambda t, m: Sample(targets=t, meta=m)
        else:
            output_signature = (
                data.dataset_input_layer_specs(),
                data.dataset_target_layer_specs(),
                data.dataset_meta_layer_specs(),
            )
            extract = lambda s: (s.inputs, s.targets, pack_meta(s.meta))
            to_sample = lambda i, t, m: Sample(inputs=i, targets=t, meta=m)

        flat_samples = []
        for sample in samples:
            sample = extract(sample)
            tf.nest.assert_same_structure(sample, output_signature)
            flat_samples.append(tf.nest.flatten(sample))

        def default(dtype):
            if dtype == tf.bool:
                return False
            return "" if dtype == tf.string else 0

        def pad(struct):
            struct, signature = struct
            padding_value = data.padding_values().get(signature.name, default(signature.dtype))
            if signature.dtype == "string":
                return np.stack(struct, axis=0)

            # assert shapes
            for i, axis_dim in enumerate(signature.shape):
                if axis_dim is None:
                    continue
                for s in struct:
                    assert s.shape[i] == axis_dim, f"Shape mismatch. Sample shape {s.shape[i]} but must be {axis_dim}"

            # pad all None axis
            for i, axis_dim in enumerate(signature.shape):
                if axis_dim is not None:
                    continue

                max_dim = max(s.shape[i] for s in struct)

                def pad_shape_for_sample(s):
                    shape = []
                    for i_ax, ax in enumerate(s.shape):
                        if i_ax == i:
                            shape.append((0, max_dim - ax))
                        else:
                            shape.append((0, 0))
                    return shape

                struct = [np.pad(s, pad_shape_for_sample(s), constant_values=padding_value) for s in struct]

            struct = np.stack(struct, axis=0)
            return struct

        flat_signature = tf.nest.flatten(output_signature)
        batched_samples = zip(*flat_samples)
        batched = list(map(pad, zip(batched_samples, flat_signature)))
        batched = tf.nest.pack_sequence_as(output_signature, batched)
        return to_sample(*batched)


def _wrap_padded_batch(
    mode: PipelineMode, dataset: "tf.data.Dataset", pipeline_params: DataPipelineParams, data: DataBase
) -> "tf.data.Dataset":
    """
    Pad and batch a tf.data.Dataset
    """
    pad_values = data.padding_values()

    def default(dtype):
        if dtype == tf.bool:
            return False
        return "" if dtype == tf.string else 0

    meta_shapes = {k: v.shape for k, v in data.meta_layer_specs().items()}
    meta_values = {
        k: tf.constant(pad_values.get(k, default(v.dtype)), dtype=v.dtype) for k, v in data.meta_layer_specs().items()
    }
    if mode == PipelineMode.PREDICTION:
        shapes = ({k: v.shape for k, v in data.input_layer_specs().items()}, meta_shapes)
        values = (
            {
                k: tf.constant(pad_values.get(k, default(v.dtype)), dtype=v.dtype)
                for k, v in data.input_layer_specs().items()
            },
            meta_values,
        )
    elif mode == PipelineMode.TARGETS:
        shapes = ({k: v.shape for k, v in data.target_layer_specs().items()}, meta_shapes)
        values = (
            {
                k: tf.constant(pad_values.get(k, default(v.dtype)), dtype=v.dtype)
                for k, v in data.target_layer_specs().items()
            },
            meta_values,
        )
    else:
        shapes = (
            {k: v.shape for k, v in data.input_layer_specs().items()},
            {k: v.shape for k, v in data.target_layer_specs().items()},
            meta_shapes,
        )
        values = (
            {
                k: tf.constant(pad_values.get(k, default(v.dtype)), dtype=v.dtype)
                for k, v in data.input_layer_specs().items()
            },
            {
                k: tf.constant(pad_values.get(k, default(v.dtype)), dtype=v.dtype)
                for k, v in data.target_layer_specs().items()
            },
            meta_values,
        )

    if pipeline_params.bucket_boundaries:
        data_elem_len_fn = data.element_length_fn()

        def element_length_func(*args):
            inputs = args[0]
            return data_elem_len_fn(inputs)

        if pipeline_params.bucket_batch_sizes:
            bucket_batch_sizes = pipeline_params.bucket_batch_sizes
        else:
            # no batch sizes given, so automatically compute them so that roughly the same amount of "tokens"
            # is in a batch. The batch size refers to the batch size for the largest bucket
            max_len = max(pipeline_params.bucket_boundaries)
            bucket_batch_sizes = [
                max(1, (pipeline_params.batch_size * size) // max_len) for size in pipeline_params.bucket_boundaries
            ]
            bucket_batch_sizes.append(bucket_batch_sizes[-1])  # for sizes larger than the upper bucked boundary

        return dataset.apply(
            bucket_by_sequence_length(
                element_length_func=element_length_func,
                bucket_batch_sizes=bucket_batch_sizes,
                bucket_boundaries=pipeline_params.bucket_boundaries,
                padded_shapes=shapes,
                padding_values=values,
                drop_remainder=pipeline_params.batch_drop_remainder,
            )
        )
    else:
        return dataset.padded_batch(
            pipeline_params.batch_size, shapes, values, drop_remainder=pipeline_params.batch_drop_remainder
        )


def _wrap_dataset(
    mode: PipelineMode,
    dataset: tf.data.Dataset,
    pipeline_params: DataPipelineParams,
    data: DataBase,
    yields_batches: bool,
) -> tf.data.Dataset:
    """
    Shuffle, pad, batch, and prefetch a tf.data.Dataset
    """
    if pipeline_params.shuffle_buffer_size > 1:
        dataset = dataset.shuffle(pipeline_params.shuffle_buffer_size)
    if not yields_batches:
        dataset = _wrap_padded_batch(mode, dataset, pipeline_params, data)
    if pipeline_params.prefetch > 0:
        dataset = dataset.prefetch(pipeline_params.prefetch)
    dataset = dataset.take(compute_limit(pipeline_params.limit, pipeline_params.batch_size))
    return dataset
