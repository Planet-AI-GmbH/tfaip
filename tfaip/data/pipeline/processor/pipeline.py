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
"""Implementation of the actual pipelines that run the DataProcessors optionally in parallel"""
import logging
from functools import partial
from typing import List, Iterable, TYPE_CHECKING

from tfaip import DataBaseParams
from tfaip import PipelineMode, Sample
from tfaip.data.pipeline.processor.dataprocessor import DataProcessorParams, GeneratingDataProcessor, SequenceProcessor, \
    MappingDataProcessor
from tfaip.data.pipeline.processor.params import ComposedProcessorPipelineParams
from tfaip.data.pipeline.processor.sample.processorpipeline import SampleProcessorPipelineBase, \
    ParallelMappingSampleProcessingPipeline, MappingSampleProcessorPipeline, ParallelGeneratingSampleProcessorPipeline, \
    GeneratingSampleProcessorPipeline

if TYPE_CHECKING:
    from tfaip.data.pipeline.datapipeline import DataPipeline

logger = logging.getLogger(__name__)


def _create_sequence_processor_fn(proc_params: List[DataProcessorParams], data_params: 'DataBaseParams',
                                  mode: PipelineMode) -> SequenceProcessor:
    processors: List[MappingDataProcessor] = []
    for p in proc_params:
        proc = p.create(data_params, mode)
        if proc is None:
            continue
        if not isinstance(proc, MappingDataProcessor):
            raise TypeError('Only MappingDataProcessors allowed')
        processors.append(proc)
    return SequenceProcessor(data_params, mode, processors)


def _create_generator_processor_fn(proc_params: DataProcessorParams, data_params: 'DataBaseParams',
                                   mode: PipelineMode) -> GeneratingDataProcessor:
    assert issubclass(proc_params.cls(), GeneratingDataProcessor), 'Only valid for GeneratingDataProcessors'
    return proc_params.create(data_params, mode)


class DataProcessorPipeline:
    """
    The actual DataProcessorPipeline which should be created by calling DataProcessorPipeline.from_params
    """

    @staticmethod
    def from_params(
            data_pipeline: 'DataPipeline',
            pipeline_params: ComposedProcessorPipelineParams,
            data_params: DataBaseParams,
            mode: PipelineMode) -> 'DataProcessorPipeline':
        # convert to pipelines
        # The asserts check that the input pipeline_params are valid
        sample_processor_pipelines: List[SampleProcessorPipelineBase] = []
        for pipeline in pipeline_params.pipelines:
            data_processors = []
            for p in pipeline.processors:
                if mode in p.modes:
                    data_processors.append(p)
                else:
                    logger.debug('{} should not be created since the pipeline mode {} is not in its modes {}',
                                 p.__class__.__name__, mode, [m.value for m in p.modes])

            if len(data_processors) == 0:
                continue
            processor_type = data_processors[0].cls()
            if issubclass(processor_type, MappingDataProcessor):
                # All Processors must be MappingDataProcessors if the first one is a MappingDataProcessor
                assert all(issubclass(dp.cls(), MappingDataProcessor) for dp in data_processors)
                cfn = partial(_create_sequence_processor_fn, data_processors, data_params, mode)
                if pipeline.run_parallel and len(data_processors) > 0:
                    sample_processor_pipelines.append(
                        ParallelMappingSampleProcessingPipeline(pipeline, data_pipeline, cfn))
                else:
                    sample_processor_pipelines.append(MappingSampleProcessorPipeline(pipeline, data_pipeline, cfn))
            elif issubclass(processor_type, GeneratingDataProcessor):
                # Only one GeneratingDataProcessor per SequenceProcessorParams
                assert all(issubclass(dp.cls(), GeneratingDataProcessor) for dp in data_processors)
                assert len(data_processors) == 1
                cfn = partial(_create_generator_processor_fn, data_processors[0], data_params, mode)
                if pipeline.run_parallel:
                    sample_processor_pipelines.append(
                        ParallelGeneratingSampleProcessorPipeline(pipeline, data_pipeline, cfn))
                else:
                    sample_processor_pipelines.append(GeneratingSampleProcessorPipeline(pipeline, data_pipeline, cfn))
            else:
                raise NotImplementedError

        return DataProcessorPipeline(sample_processor_pipelines)

    def __init__(self, pipeline: List[SampleProcessorPipelineBase]):
        self.pipeline = pipeline

    def apply(self, samples: Iterable[Sample]) -> Iterable[Sample]:
        """
        Apply the individual pipelines (Mapping or Generating, possibly parallel) on the samples
        """
        for p in self.pipeline:
            samples = p.apply(samples)

        return samples

    def apply_on_sample(self, sample: Sample) -> Sample:
        """
        Apply the individual pipelines (Mapping or Generating, possibly parallel) on the samples
        Use this with caution since all pipelines are created first which is a great overhead if only one sample is
        processed.
        In this case, usually `run_parallel` should be set to false to remove at least the biggest part of the overhead.
        Running in parallel is not required in this case (usually).
        """
        return next(iter(self.apply([sample])))
