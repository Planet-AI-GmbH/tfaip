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
from typing import TYPE_CHECKING, Callable, Optional

from tfaip.base.data.pipeline.dataprocessor import SequenceProcessor
from tfaip.base.data.pipeline.definitions import PipelineMode
from tfaip.util.multiprocessing.data.worker import DataWorker

if TYPE_CHECKING:
    from tfaip.base.data.databaseparams import DataBaseParams


class PreprocWorker(DataWorker):
    def __init__(self,
                 params: 'DataBaseParams',
                 mode: PipelineMode,
                 data_processor_fn: Callable[[], SequenceProcessor],
                 ):
        self.params = params
        self.mode = mode
        self.data_processor_fn = data_processor_fn
        self.processors: Optional[SequenceProcessor] = None

    def initialize_thread(self):
        self.processors = self.data_processor_fn()

    def process(self, *args, **kwargs):
        return self.processors.apply_on_sample(args[0])
