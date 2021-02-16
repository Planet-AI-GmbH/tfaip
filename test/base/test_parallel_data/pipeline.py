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
from typing import Callable

from test.base.test_parallel_data.worker import Worker
from tfaip.util.multiprocessing.data.pipeline import DataPipeline
from tfaip.util.multiprocessing.data.worker import DataWorker


def create():
    return Worker()


class Pipeline(DataPipeline):
    def __init__(self, data, processes, max_int):
        super(Pipeline, self).__init__(data, processes)
        self.max_int = max_int

    def create_worker_func(self) -> Callable[[], DataWorker]:
        return create

    def generate_input(self):
        return range(self.max_int)
