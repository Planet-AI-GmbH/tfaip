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
"""Different version of executing DataProcessors."""
from abc import abstractmethod, ABC
from typing import Optional, Callable, Iterable, TYPE_CHECKING, Union

from tfaip import Sample, PipelineMode
from tfaip.data.pipeline.processor.dataprocessor import SequenceProcessor, GeneratingDataProcessor
from tfaip.data.pipeline.processor.params import SequentialProcessorPipelineParams
from tfaip.data.pipeline.processor.sample.parallelgenerator import ParallelDataGenerator
from tfaip.data.pipeline.processor.sample.parallelpipeline import ParallelDataProcessorPipeline

if TYPE_CHECKING:
    from tfaip.data.pipeline.datapipeline import DataPipeline


class SampleProcessorPipelineBase(ABC):
    """
    Base Pipeline that instantiates its DataProcessors in a (optionally) separate thread.
    """

    def __init__(self,
                 params: SequentialProcessorPipelineParams,
                 data_pipeline: 'DataPipeline',
                 processor_fn: Optional[Callable[[], Union[SequenceProcessor, GeneratingDataProcessor]]] = None):
        self.params = params
        self.data_pipeline = data_pipeline
        self.create_processor_fn = processor_fn

    @abstractmethod
    def apply(self, samples: Iterable[Sample]) -> Iterable[Sample]:
        raise NotImplementedError


class MappingSampleProcessorPipeline(SampleProcessorPipelineBase):
    """
    Implementation for MappingDataProcessors
    """

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


class ParallelMappingSampleProcessingPipeline(MappingSampleProcessorPipeline):
    """
    Parallel version of the implementation for MappingDataProcessors
    """

    def apply(self, samples: Iterable[Sample]) -> Iterable[Sample]:
        if not self.params.run_parallel:
            for x in super().apply(samples):
                yield x
        else:
            parallel_pipeline = ParallelDataProcessorPipeline(
                self.data_pipeline,
                samples,
                create_processor_fn=self.create_processor_fn,
                auto_repeat_input=False,
                preproc_max_tasks_per_child=self.params.max_tasks_per_process,
                num_processes=self.params.num_threads if self.params.num_threads >= 1 else None,
            )
            for x in parallel_pipeline.output_generator():
                yield x

            parallel_pipeline.join()


class GeneratingSampleProcessorPipeline(SampleProcessorPipelineBase):
    """
    Implementation for GeneratingDataProcessors.
    """

    def apply(self, samples: Iterable[Sample]) -> Iterable[Sample]:
        if not self.create_processor_fn:
            for sample in samples:
                yield sample
        else:
            processor: GeneratingDataProcessor = self.create_processor_fn()
            for sample in processor.generate(samples):
                yield sample


class ParallelGeneratingSampleProcessorPipeline(GeneratingSampleProcessorPipeline):
    """
    Parallel version of the implementation for GeneratingDataProcessors.
    Note: Only use if PipelineMode is Training, since the outputs are not ordered!
    """

    def apply(self, samples: Iterable[Sample]) -> Iterable[Sample]:
        if not self.params.run_parallel:
            for x in super().apply(samples):
                yield x
        else:
            num_threads = self.params.num_threads if self.data_pipeline.mode == PipelineMode.TRAINING else 1
            # If not training, enforce num threads as 1 to yield deterministic results
            with ParallelDataGenerator(
                    self.data_pipeline,
                    samples,
                    create_processor_fn=self.create_processor_fn,
                    auto_repeat_input=False,
                    preproc_max_tasks_per_child=self.params.max_tasks_per_process,
                    num_processes=num_threads if num_threads >= 1 else None,
            ) as parallel_pipeline:
                for x in parallel_pipeline.output_generator():
                    yield x
