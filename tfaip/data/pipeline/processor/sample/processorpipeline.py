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
from typing import Optional, Callable, Iterable, TYPE_CHECKING, Union, List

from tfaip import Sample, PipelineMode
from tfaip.data.pipeline.processor.dataprocessor import SequenceProcessor, GeneratingDataProcessor
from tfaip.data.pipeline.processor.params import SequentialProcessorPipelineParams
from tfaip.data.pipeline.processor.sample.parallelgenerator import ParallelDataGenerator
from tfaip.data.pipeline.processor.sample.parallelpipeline import ParallelDataProcessorPipeline

if TYPE_CHECKING:
    from tfaip.data.pipeline.datapipeline import DataPipelineParams


class SampleProcessorPipelineBase(ABC):
    """
    Base Pipeline that instantiates its DataProcessors in a (optionally) separate thread.
    """

    def __init__(
        self,
        params: SequentialProcessorPipelineParams,
        data_pipeline_params: "DataPipelineParams",
        processor_fn: Optional[Callable[[], Union[SequenceProcessor, GeneratingDataProcessor]]] = None,
    ):
        self.params = params
        self.data_pipeline_params = data_pipeline_params
        self.create_processor_fn = processor_fn

    def apply(self, samples: Iterable[Sample], run_parallel: Optional[bool] = None) -> Iterable[Sample]:
        """Apply the data processors on the samples

        This will create the processing pipeline, process all samples and destroy the pipeline.
        Use this on as many samples as possible to reduce the time for initialization/destruction.

        Params:
            run_parallel: override the run_parallel setting of the params
        """
        if run_parallel is None:
            run_parallel = self.params.run_parallel
        return self._apply(samples, run_parallel)

    @abstractmethod
    def _apply(self, samples: Iterable[Sample], run_parallel) -> Iterable[Sample]:
        raise NotImplementedError


class MappingSampleProcessorPipeline(SampleProcessorPipelineBase):
    """
    Implementation for MappingDataProcessors
    """

    def _apply(
        self, samples: Iterable[Union[Sample, List[Sample]]], run_parallel
    ) -> Iterable[Union[Sample, List[Sample]]]:
        if not self.create_processor_fn:
            for sample in samples:
                yield sample
        else:
            processor = self.create_processor_fn()
            for sample in samples:
                if isinstance(sample, list):
                    # batched input
                    r = list(filter(lambda x: x is not None, map(processor.apply_on_sample, sample)))
                    if len(r) > 0:
                        yield r
                else:
                    r = processor.apply_on_sample(sample)
                    if r is not None:
                        yield r


class ParallelMappingSampleProcessingPipeline(MappingSampleProcessorPipeline):
    """
    Parallel version of the implementation for MappingDataProcessors
    """

    def _apply(self, samples: Iterable[Sample], run_parallel) -> Iterable[Sample]:
        if not run_parallel:
            for x in super().apply(samples):
                yield x
        else:
            parallel_pipeline = ParallelDataProcessorPipeline(
                self.data_pipeline_params,
                samples,
                create_processor_fn=self.create_processor_fn,
                auto_repeat_input=False,
                preproc_max_tasks_per_child=self.params.max_tasks_per_process,
                num_processes=self.params.num_threads if self.params.num_threads >= 1 else None,
            )
            with parallel_pipeline as output_generator:
                for x in output_generator:
                    yield x


class GeneratingSampleProcessorPipeline(SampleProcessorPipelineBase):
    """
    Implementation for GeneratingDataProcessors.
    """

    def _apply(self, samples: Iterable[Sample], run_parallel) -> Iterable[Sample]:
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

    def _apply(self, samples: Iterable[Sample], run_parallel) -> Iterable[Sample]:
        if not run_parallel:
            for x in super().apply(samples):
                yield x
        else:
            num_threads = self.params.num_threads if self.data_pipeline_params.mode == PipelineMode.TRAINING else 1
            # If not training, enforce num threads as 1 to yield deterministic results
            with ParallelDataGenerator(
                self.data_pipeline_params,
                samples,
                create_processor_fn=self.create_processor_fn,
                auto_repeat_input=False,
                preproc_max_tasks_per_child=self.params.max_tasks_per_process,
                num_processes=num_threads if num_threads >= 1 else None,
            ) as output_generator:
                for x in output_generator:
                    yield x
