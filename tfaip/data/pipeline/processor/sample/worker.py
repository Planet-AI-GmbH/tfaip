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
"""Definitions of the workers for DataProcessors that are actually run in a separate process"""
from typing import TYPE_CHECKING, Callable, Optional, Iterator

from tfaip import PipelineMode, Sample
from tfaip.data.pipeline.processor.dataprocessor import SequenceProcessor, GeneratingDataProcessor
from tfaip.util.multiprocessing.data.worker import DataWorker

if TYPE_CHECKING:
    from tfaip.data.databaseparams import DataBaseParams


class MappingDataProcessorWorker(DataWorker):
    """Worker that applies a SequenceProcessor

    i.e., a list of MappingDataProcessors. The worker will be instantiated and called in a separate process.
    """

    def __init__(
        self,
        data_processor_fn: Callable[[], SequenceProcessor],
    ):
        self.data_processor_fn = data_processor_fn
        self.processors: Optional[SequenceProcessor] = None

    def initialize_thread(self):
        self.processors = self.data_processor_fn()

    def process(self, *args, **kwargs):
        sample = args[0]
        if isinstance(sample, list):
            return [self.processors.apply_on_sample(s) for s in sample]
        else:
            return self.processors.apply_on_sample(sample)


class GeneratingDataProcessorWorker(DataWorker):
    """Worker that applies one GeneratingDataProcessor

    The worker will be instantiated and called in a separate process.
    """

    def __init__(
        self,
        data_processor_fn: Callable[[], GeneratingDataProcessor],
    ):
        self.data_processor_fn = data_processor_fn
        self.processor: Optional[GeneratingDataProcessor] = None

    def initialize_thread(self):
        self.processor = self.data_processor_fn()

    def process(self, sample: Iterator[Sample]):
        return self.processor.generate(sample)
