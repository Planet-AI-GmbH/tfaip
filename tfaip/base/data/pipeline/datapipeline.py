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
import copy
import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, List, Iterable, Optional, Callable, Type
import logging

from dataclasses_json import dataclass_json

from tfaip.base.data.pipeline.dataprocessor import DataProcessorFactory, SequenceProcessor, DataProcessor
from tfaip.base.data.pipeline.definitions import Sample, PipelineMode, DataProcessorFactoryParams, \
    GENERAL_PROCESSOR
from tfaip.base.data.pipeline.parallelpipeline import ParallelDataProcessorPipeline
from tfaip.base.data.pipeline.tfdatasetgenerator import TFDatasetGenerator
from tfaip.util.multiprocessing.join import JoinableHolder

if TYPE_CHECKING:
    from tfaip.base.data.data import DataBase
    from tfaip.base.data.databaseparams import DataGeneratorParams


logger = logging.getLogger(__name__)


class SampleConsumer:
    pass


def create_processor_fn(factory: DataProcessorFactory, processors: List[DataProcessorFactoryParams], params, mode: PipelineMode) -> SequenceProcessor:
    return factory.create_sequence(processors, params, mode)


@dataclass_json
@dataclass
class SamplePipelineParams:
    run_parallel: bool = True
    sample_processors: List[DataProcessorFactoryParams] = field(default_factory=list)


class SampleProcessorPipeline:
    def __init__(self, data_pipeline: 'DataPipeline', processor_fn: Optional[Callable[[], SequenceProcessor]] = None):
        self.data_pipeline = data_pipeline
        self.create_processor_fn = processor_fn

    def apply(self, samples: Iterable[Sample]) -> Iterable[Sample]:
        if not self.create_processor_fn:
            for sample in samples:
                yield sample
        else:
            processor = self.create_processor_fn()
            for sample in samples:
                r = processor.apply_on_sample(sample)
                if r is not None:
                    yield r


class ParallelSampleProcessingPipeline(SampleProcessorPipeline):
    def apply(self, samples: Iterable[Sample]) -> Iterable[Sample]:
        parallel_pipeline = ParallelDataProcessorPipeline(self.data_pipeline, samples,
                                                          create_processor_fn=self.create_processor_fn,
                                                          auto_repeat_input=False)
        for x in parallel_pipeline.output_generator():
            yield x

        parallel_pipeline.join()


class DataGenerator(ABC):
    def __init__(self, mode: PipelineMode, params: 'DataGeneratorParams'):
        params.validate()
        self.mode = mode
        self.params = params

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def generate(self) -> Iterable[Sample]:
        raise NotImplementedError


class RawDataGenerator(DataGenerator):
    def __init__(self, raw_data: List[Sample], mode: PipelineMode, params: 'DataGeneratorParams'):
        super(RawDataGenerator, self).__init__(mode, params)
        self.raw_data = raw_data

    def __len__(self):
        return len(self.raw_data)

    def generate(self) -> Iterable[Sample]:
        return self.raw_data


def _create_sequence_processor_fn(factory, *args) -> Callable[[], SequenceProcessor]:
    return factory.create_sequence(*args)


class DataPipeline(JoinableHolder, ABC):
    def _sequence_processor_fn(self, params: SamplePipelineParams) -> Callable[[], SequenceProcessor]:
        factory = self.data.__class__.data_processor_factory()
        data_params = self.data_params
        mode = self.mode
        return partial(_create_sequence_processor_fn, factory, params.sample_processors, data_params, mode)

    def __init__(self,
                 mode: PipelineMode,
                 data_base: 'DataBase',
                 generator_params: 'DataGeneratorParams',
                 input_processors=None,
                 output_processors=None,
                 ):
        super(DataPipeline, self).__init__()
        self.generator_params = generator_params
        self.data = data_base
        self.data_params = data_base.params()
        self.mode = mode
        self._auto_batch = True
        self._input_processors: SamplePipelineParams = input_processors or self.data_params.pre_processors_
        self._output_processors: SamplePipelineParams = output_processors or self.data_params.post_processors_

    def to_mode(self, mode: PipelineMode) -> 'DataPipeline':
        return self.__class__(mode, self.data, self.generator_params, self._input_processors, self._output_processors)

    def as_preloaded(self, progress_bar=True) -> 'RawDataPipeline':
        logger.info(f"Preloading: Converting {self.mode.value} to raw pipeline.")
        with self as rp:
            non_preloadable_params = []
            data = rp.preload_input_samples(progress_bar=progress_bar, non_preloadable_params=non_preloadable_params)
            pipeline = RawDataPipeline(data, self.mode, self.data, self.generator_params)
            pipeline._input_processors = copy.copy(pipeline._input_processors)
            pipeline._input_processors.sample_processors = non_preloadable_params
            return pipeline

    @property
    def auto_batch(self):
        return self._auto_batch

    @abstractmethod
    def create_data_generator(self) -> DataGenerator:
        raise NotImplementedError

    def flat_input_processors(self, preload=False, non_preloadable_params=[]) -> List[DataProcessor]:
        factory = self.data.__class__.data_processor_factory()
        params: SamplePipelineParams = self._input_processors

        if params:
            processors = [factory.create(sp, self.data_params, self.mode) for sp in params.sample_processors]
            out_processors = []
            i = 0
            for i, p in enumerate(processors, start=1):
                if p is None:
                    continue  # processor not available in mode
                if preload and not p.supports_preload():
                    i -= 1
                    break
                out_processors.append(p)
            non_preloadable_params.extend(params.sample_processors[i:])
            processors = out_processors
        else:
            processors = []

        return processors

    def create_input_pipeline(self) -> Optional[SampleProcessorPipeline]:
        params: SamplePipelineParams = self._input_processors

        def sample_processor_class(p: SamplePipelineParams) -> Type[SampleProcessorPipeline]:
            if p.run_parallel and len(p.sample_processors) > 0:
                return ParallelSampleProcessingPipeline
            return SampleProcessorPipeline

        if params:
            return sample_processor_class(params)(self, self._sequence_processor_fn(params))
        else:
            return SampleProcessorPipeline(self)

    def create_tf_dataset_generator(self) -> TFDatasetGenerator:
        return TFDatasetGenerator(self)

    def create_output_pipeline(self) -> Optional[SampleProcessorPipeline]:
        params: SamplePipelineParams = self._output_processors
        if params:
            if params.run_parallel and len(params.sample_processors) > 0:
                return ParallelSampleProcessingPipeline(self, self._sequence_processor_fn(params))
            else:
                return SampleProcessorPipeline(self, self._sequence_processor_fn(params))
        return SampleProcessorPipeline(self)

    def create_data_consumer(self) -> SampleConsumer:
        return SampleConsumer()

    def __enter__(self):
        from tfaip.base.data.pipeline.runningdatapipeline import RunningDataPipeline
        return RunningDataPipeline(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.join()
        gc.collect()  # required or something goes wrong with tf.data.Dataset


class SimpleDataPipeline(DataPipeline, ABC):
    @abstractmethod
    def generate_samples(self) -> Iterable[Sample]:
        raise NotImplementedError

    def number_of_samples(self) -> int:
        return 1

    def create_data_generator(self) -> DataGenerator:
        generator = self.generate_samples()
        number_of_samples = self.number_of_samples()

        class SimpleDataGenerator(DataGenerator):
            def __len__(self):
                return number_of_samples

            def generate(self) -> Iterable[Sample]:
                return generator

        return SimpleDataGenerator(self.mode, self.generator_params)


class RawDataPipeline(DataPipeline):
    def __init__(self,
                 samples: List[Sample],
                 mode: PipelineMode,
                 data_base: 'DataBase',
                 generator_params: 'DataGeneratorParams',
                 input_processors=None,
                 output_processors=None,
                 ):
        super(RawDataPipeline, self).__init__(mode, data_base, generator_params, input_processors, output_processors)
        self.samples = samples


    def to_mode(self, mode: PipelineMode) -> 'DataPipeline':
        return self.__class__(self.samples, mode, self.data, self.generator_params, self._input_processors, self._output_processors)

    def create_data_generator(self) -> DataGenerator:
        return RawDataGenerator(self.samples, self.mode, self.generator_params)
