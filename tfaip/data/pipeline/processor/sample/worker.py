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

    The generating data processor worker support a "pre-/post-processing sequential processor" which will be
    executed in the same thread.

    The worker will be instantiated and called in a separate process.
    """

    def __init__(
        self,
        pre_mapping_processor_fn: Optional[Callable[[], SequenceProcessor]],
        generating_processor_fn: Callable[[], GeneratingDataProcessor],
        post_mapping_processor_fn: Optional[Callable[[], SequenceProcessor]],
    ):
        self.pre_mapping_data_processor_fn = pre_mapping_processor_fn
        self.post_mapping_data_processor_fn = post_mapping_processor_fn
        self.generating_data_processor_fn = generating_processor_fn
        self.pre_mapping_processor: Optional[SequenceProcessor] = None
        self.post_mapping_processor: Optional[SequenceProcessor] = None
        self.generating_processor: Optional[GeneratingDataProcessor] = None

    def initialize_thread(self):
        if self.pre_mapping_data_processor_fn:
            self.pre_mapping_processor = self.pre_mapping_data_processor_fn()
        self.generating_processor = self.generating_data_processor_fn()
        if self.post_mapping_data_processor_fn:
            self.post_mapping_processor = self.post_mapping_data_processor_fn()

    def process(self, sample: Iterator[Sample]):
        if self.pre_mapping_processor:
            sample = map(self.pre_mapping_processor.apply_on_sample, sample)
        sample = self.generating_processor.generate(sample)
        if self.post_mapping_processor:
            sample = map(self.post_mapping_processor.apply_on_sample, sample)
        return sample
