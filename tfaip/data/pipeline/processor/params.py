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
"""Definition of the DataProcessorPipelineParams,

Included are the definition of its derived SequentialProcessorPipelineParams and ComposedProcessorPipelineParams.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING, TypeVar, Type

from paiargparse import pai_dataclass, pai_meta

from tfaip.data.pipeline.processor.dataprocessor import DataProcessorParams

if TYPE_CHECKING:
    from tfaip import DataBaseParams
    from tfaip.data.pipeline.datapipeline import DataPipeline, DataPipelineParams
    from tfaip.data.pipeline.processor.pipeline import DataProcessorPipeline

TP = TypeVar("TP", bound=DataProcessorParams)


@pai_dataclass
@dataclass
class DataProcessorPipelineParams(ABC):
    """
    General parameters for a `DataProcessorPipeline`.
    The parameters provide a function to create the respective Pipeline (`create`)
    """

    run_parallel: bool = field(default=True, metadata=pai_meta(help="Run this pipeline in parallel."))
    num_threads: int = field(
        default=-1,
        metadata=pai_meta(
            help="The number of threads to use for this pipeline. Else use the value of the generator params."
        ),
    )  # Upper limit for the threads to use maximal (default = -1 = no limit)
    max_tasks_per_process: int = field(
        default=250,
        metadata=pai_meta(
            help="Maximum tasks of a child in the preproc pipeline after a child is recreated. "
            "Higher numbers for better performance but on the drawback if higher memory consumption. "
            "Only used if the scenario uses a DataPipeline."
        ),
    )

    def create_with_pipeline(self, data_pipeline: "DataPipeline") -> "DataProcessorPipeline":
        return self.create(data_pipeline.pipeline_params, data_pipeline.data.params)

    @abstractmethod
    def create(
        self, data_pipeline_params: "DataPipelineParams", data_params: "DataBaseParams"
    ) -> "DataProcessorPipeline":
        """
        Create the actual DataProcessorPipeline
        """
        raise NotImplementedError

    def has_processor(self, t: Type[DataProcessorParams]) -> bool:
        """
        Check if a DataProcessor of type `t` is present anywhere in the pipeline
        """
        return len(self.processors_of_type(t)) > 0

    @abstractmethod
    def flatten_processors(self) -> List[DataProcessorParams]:
        """
        Retrieve a flattened sequence of all DataProcessorParams
        """
        raise NotImplementedError

    def processors_of_type(self, t: Type[TP]) -> List[TP]:
        """
        Retrieve a sequence of all DataProcessorParams matching the type `t`
        """
        return list(filter(lambda p: issubclass(p.__class__, t), self.flatten_processors()))

    @abstractmethod
    def erase_all(self, t: Type[DataProcessorParams]):
        """
        Erase all DataProcessorParams matching the type `t`
        """
        raise NotImplementedError

    @abstractmethod
    def replace_all(self, t: Type[DataProcessorParams], p: DataProcessorParams):
        """
        Replace all DataProcessorParams matching the type `t` with `p`
        """
        raise NotImplementedError


@pai_dataclass
@dataclass
class SequentialProcessorPipelineParams(DataProcessorPipelineParams):
    """
    The SequentialProcessorPipelineParams have a flat list of DataProcessorParams of any type (Mapping or Generating)
    Upon creation, the processors will automatically be grouped to Processors of same type which will then be used to
    create the actual ComposedProcessorPipeline.
    """

    processors: List[DataProcessorParams] = field(default_factory=list)

    def flatten_processors(self) -> List[DataProcessorParams]:
        return self.processors

    def erase_all(self, t: Type[DataProcessorParams]):
        self.processors = list(filter(lambda p: not isinstance(p, t), self.processors))

    def replace_all(self, t: Type[DataProcessorParams], p: DataProcessorParams):
        self.processors = [p if isinstance(sp, t) else sp for sp in self.processors]

    def create(
        self, data_pipeline_params: "DataPipelineParams", data_params: "DataBaseParams"
    ) -> "DataProcessorPipeline":
        from tfaip.data.pipeline.processor.dataprocessor import (
            MappingDataProcessor,
            GeneratingDataProcessor,
            SequenceProcessor,
        )  # pylint: disable=import-outside-toplevel

        composed_params = ComposedProcessorPipelineParams(pipelines=[])
        last_type = None
        # Group parameters into pipelines
        for p in self.processors:
            processor_type = p.cls()
            if issubclass(processor_type, MappingDataProcessor):
                if len(composed_params.pipelines) == 0 or last_type != SequenceProcessor:
                    composed_params.pipelines.append(
                        SequentialProcessorPipelineParams(
                            run_parallel=self.run_parallel,
                            num_threads=self.num_threads,
                            max_tasks_per_process=self.max_tasks_per_process,
                            processors=[p],
                        )
                    )
                else:
                    composed_params.pipelines[-1].processors.append(p)

                last_type = SequenceProcessor
            elif issubclass(processor_type, GeneratingDataProcessor):
                composed_params.pipelines.append(
                    SequentialProcessorPipelineParams(
                        run_parallel=self.run_parallel,
                        num_threads=self.num_threads,
                        max_tasks_per_process=self.max_tasks_per_process,
                        processors=[p],
                    )
                )
                last_type = GeneratingDataProcessor
            else:
                raise NotImplementedError

        return composed_params.create(data_pipeline_params, data_params)


@pai_dataclass
@dataclass
class ComposedProcessorPipelineParams(DataProcessorPipelineParams):
    """
    The ComposedProcessorPipelineParams can be used to define a list of SequentialProcessorPipelineParams.
    Note that within a SequentialProcessorPipeline only one GeneratingDataProcessorParams may exists, or a list of
    MappingDataProcessorParams.
    """

    pipelines: List[SequentialProcessorPipelineParams] = field(default_factory=list)

    def erase_all(self, t: Type[DataProcessorParams]):
        for p in self.pipelines:
            p.erase_all(t)

    def replace_all(self, t: Type[DataProcessorParams], p: DataProcessorParams):
        for pipeline in self.pipelines:
            pipeline.replace_all(t, p)

    def flatten_processors(self) -> List[DataProcessorParams]:
        return sum((p.flatten_processors() for p in self.pipelines), [])

    def create(
        self, data_pipeline_params: "DataPipelineParams", data_params: "DataBaseParams"
    ) -> "DataProcessorPipeline":
        from tfaip.data.pipeline.processor.pipeline import (
            DataProcessorPipeline,
        )  # pylint: disable=import-outside-toplevel

        return DataProcessorPipeline.from_params(data_pipeline_params, self, data_params)
