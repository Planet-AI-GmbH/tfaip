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
from typing import TYPE_CHECKING, Iterable, List, Optional

import numpy as np
import tensorflow as tf

from tfaip.base.data.pipeline.definitions import Sample, PipelineMode
from tfaip.util.multiprocessing.parallelmap import tqdm_wrapper

if TYPE_CHECKING:
    from tfaip.base.data.pipeline.datapipeline import DataPipeline


def compute_limit(limit, batch_size):
    assert(limit != 0)
    if limit < 0:
        return limit  # no limit
    else:
        return -(-limit // batch_size)  # ceiled integer div => 1 // 3 = 1; 3 // 3 => 1; 4 // 3 = 2


class RunningDataPipeline:
    def __init__(self, data_pipeline: 'DataPipeline'):
        self.data_pipeline = data_pipeline
        self.mode = data_pipeline.mode
        self.data_generator = self.data_pipeline.create_data_generator()

    def __len__(self):
        return len(self.data_generator)

    def process_output(self, samples: Iterable[Sample]) -> Iterable[Sample]:
        output_pipeline = self.data_pipeline.create_output_pipeline()

        def extract_meta(sample: Sample) -> Sample:
            meta = sample.meta or {}
            if 'meta' in sample.inputs:
                input_meta = sample.inputs['meta']
                if isinstance(input_meta, list) or isinstance(input_meta, np.ndarray):
                    assert(len(input_meta) == 1)
                    input_meta = input_meta[0]

                meta.update(**json.loads(input_meta))
            return sample.new_meta(meta)

        samples = map(extract_meta, samples)

        if output_pipeline:
            return output_pipeline.apply(samples)
        return samples

    def input_dataset(self, auto_repeat=None) -> Optional['tf.data.Dataset']:
        if len(self) == 0:
            # Empty set
            return None

        tf_dataset_generator = self.data_pipeline.create_tf_dataset_generator()
        dataset = tf_dataset_generator.create(lambda: self.generate_input_samples(auto_repeat))
        return self._wrap_dataset(dataset)

    def preload_input_samples(self, progress_bar=True, non_preloadable_params=[]) -> List[Sample]:
        data_generator = self.data_generator
        old_limit = data_generator.params.limit
        data_generator.params.limit = len(data_generator)

        last_generator = list(tqdm_wrapper(data_generator.generate(), progress_bar=progress_bar,
                                           total=len(data_generator), desc="Loading samples"))
        processors = self.data_pipeline.flat_input_processors(preload=True, non_preloadable_params=non_preloadable_params)
        for processor in processors:
            last_generator = processor.preload(last_generator,
                                               num_processes=data_generator.params.num_processes,
                                               progress_bar=progress_bar,
                                               )
            last_generator = list(last_generator)

        data_generator.params.limit = old_limit
        return last_generator

    def generate_input_samples(self, auto_repeat=None) -> Iterable[Sample]:
        data_generator = self.data_generator
        if auto_repeat is None:
            auto_repeat = self.mode == PipelineMode.Training and data_generator.params.limit < 0

        input_pipeline = self.data_pipeline.create_input_pipeline()
        while True:
            generate = data_generator.generate()
            if input_pipeline:
                for s in input_pipeline.apply(generate):
                    yield s
            else:
                for s in generate:
                    yield s

            if not auto_repeat:
                break

    def _wrap_padded_batch(self, dataset: 'tf.data.Dataset'):
        generator_params = self.data_pipeline.generator_params
        data = self.data_pipeline.data
        pad_values = data.padding_values()

        def default(dtype):
            if dtype == tf.bool:
                return False
            return '' if dtype == tf.string else 0

        if self.mode == PipelineMode.Prediction:
            shapes = {k: v.shape for k, v in data.input_layer_specs().items()}
            values = {k: tf.constant(pad_values.get(k, default(v.dtype)), dtype=v.dtype) for k, v in data.input_layer_specs().items()}
        elif self.mode == PipelineMode.Targets:
            shapes = {k: v.shape for k, v in data.target_layer_specs().items()},
            values = {k: tf.constant(pad_values.get(k, default(v.dtype)), dtype=v.dtype) for k, v in data.target_layer_specs().items()},
        else:
            shapes = (
                {k: v.shape for k, v in data.input_layer_specs().items()},
                {k: v.shape for k, v in data.target_layer_specs().items()},
            )
            values = (
                {k: tf.constant(pad_values.get(k, default(v.dtype)), dtype=v.dtype) for k, v in data.input_layer_specs().items()},
                {k: tf.constant(pad_values.get(k, default(v.dtype)), dtype=v.dtype) for k, v in data.target_layer_specs().items()},
            )

        return dataset.padded_batch(generator_params.batch_size, shapes, values, drop_remainder=generator_params.batch_drop_remainder)

    def _wrap_dataset(self, dataset):
        generator_params = self.data_pipeline.generator_params
        if self.data_pipeline.auto_batch:
            dataset = self._wrap_padded_batch(dataset)
        if generator_params.prefetch > 0:
            dataset = dataset.prefetch(generator_params.prefetch)
        dataset = dataset.take(compute_limit(generator_params.limit, generator_params.batch_size))
        return dataset
