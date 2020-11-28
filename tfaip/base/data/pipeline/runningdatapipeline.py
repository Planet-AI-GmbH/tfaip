from typing import TYPE_CHECKING, Iterable, List, Optional

import tensorflow as tf

from tfaip.base.data.pipeline.definitions import InputTargetSample, PipelineMode, InputOutputSample
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

    def process_output(self, samples: Iterable[InputOutputSample]) -> Iterable[InputOutputSample]:
        output_pipeline = self.data_pipeline.create_output_pipeline()
        if output_pipeline:
            return output_pipeline.apply(samples)
        return samples

    def input_dataset(self, auto_repeat=None) -> Optional['tf.data.Dataset']:
        if len(self) == 0:
            # Empty set
            return None

        mode = self.mode
        def wrap():
            samples = self.generate_input_samples(auto_repeat)
            for sample in samples:
                if isinstance(sample, InputTargetSample):
                    if mode == PipelineMode.Prediction:
                        yield sample.inputs
                    elif mode == PipelineMode.Targets:
                        yield sample.targets
                    else:
                        yield sample.inputs, sample.targets
                else:
                    yield sample

        input_types = {k: v.dtype for k, v in self.data_pipeline.data.input_layer_specs().items()}
        target_types = {k: v.dtype for k, v in self.data_pipeline.data.target_layer_specs().items()}
        if self.mode == PipelineMode.Prediction:
            output_types = input_types
        elif self.mode == PipelineMode.Targets:
            output_types = target_types
        else:
            output_types = (input_types, target_types)

        dataset = tf.data.Dataset.from_generator(wrap, output_types=output_types)
        return self._wrap_dataset(dataset)

    def preload_input_samples(self, progress_bar=True, non_preloadable_params=[]) -> List[InputTargetSample]:
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

    def generate_input_samples(self, auto_repeat=None) -> Iterable[InputTargetSample]:
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
