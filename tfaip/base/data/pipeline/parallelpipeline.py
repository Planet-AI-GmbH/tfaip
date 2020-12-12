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
import itertools
from functools import partial
from typing import Callable, TYPE_CHECKING, Iterable

from tfaip.base.data.pipeline.definitions import Sample
from tfaip.base.data.pipeline.worker import PreprocWorker
from tfaip.util.multiprocessing.data.pipeline import ParallelPipeline
from tfaip.util.multiprocessing.data.worker import DataWorker

if TYPE_CHECKING:
    from tfaip.base.data.pipeline.datapipeline import DataPipeline


def create_preproc_worker(*args):
    return PreprocWorker(*args)


class ParallelDataProcessorPipeline(ParallelPipeline):
    def __init__(self,
                 data_pipeline: 'DataPipeline',
                 sample_generator: Iterable[Sample],
                 create_processor_fn,
                 auto_repeat_input: bool,
                 ):
        self.data_pipeline = data_pipeline
        self.sample_generator = sample_generator
        self.create_processor_fn = create_processor_fn
        super(ParallelDataProcessorPipeline, self).__init__(data_pipeline,
                                                            data_pipeline.generator_params.num_processes,
                                                            data_pipeline.generator_params.limit,
                                                            auto_repeat_input,
                                                            data_pipeline.data_params.preproc_max_tasks_per_child)

    def create_worker_func(self) -> Callable[[], DataWorker]:
        return partial(create_preproc_worker,
                       self.data_pipeline.data.params(),
                       self.data_pipeline.mode,
                       self.create_processor_fn,
                       )

    def generate_input(self):
        if self.limit > 0:
            return itertools.islice(self.sample_generator, self.limit)
        else:
            return self.sample_generator

