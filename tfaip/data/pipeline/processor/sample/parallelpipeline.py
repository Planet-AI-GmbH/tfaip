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
"""Parallel implementation to run MappingDataProcessors"""
import itertools
from functools import partial
from typing import Callable, TYPE_CHECKING, Iterable, Optional

from tfaip import DataBaseParams
from tfaip import Sample
from tfaip.data.pipeline.processor.sample.worker import MappingDataProcessorWorker
from tfaip.util.multiprocessing.data.pipeline import ParallelPipeline
from tfaip.util.multiprocessing.data.worker import DataWorker

if TYPE_CHECKING:
    from tfaip.data.pipeline.datapipeline import DataPipelineParams


def create_mapping_data_processor_worker(*args):
    return MappingDataProcessorWorker(*args)


class ParallelDataProcessorPipeline(ParallelPipeline):
    """
    Implementation of applying MappingDataProcessors in parallel by implementing the ParallelGenerator.
    """

    def __init__(
        self,
        pipeline_params: "DataPipelineParams",
        sample_generator: Iterable[Sample],
        create_processor_fn,
        auto_repeat_input: bool,
        num_processes: Optional[int] = None,
        limit: Optional[int] = None,
        preproc_max_tasks_per_child: Optional[int] = 250,
    ):
        if num_processes is None:
            num_processes = pipeline_params.num_processes
        if limit is None:
            limit = pipeline_params.limit
        self.sample_generator = sample_generator
        self.create_processor_fn = create_processor_fn
        super().__init__(num_processes, limit, auto_repeat_input, preproc_max_tasks_per_child)

    def create_worker_func(self) -> Callable[[], DataWorker]:
        return partial(
            create_mapping_data_processor_worker,
            self.create_processor_fn,
        )

    def generate_input(self):
        if self.limit > 0:
            return itertools.islice(self.sample_generator, self.limit)
        else:
            return self.sample_generator
