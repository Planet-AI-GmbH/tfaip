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
from abc import ABC, abstractmethod
import multiprocessing
from multiprocessing.pool import RUN
from typing import Callable
import time
import logging

from tfaip.util.multiprocessing.data.pool import WrappedPool
from tfaip.util.multiprocessing.data.worker import DataWorker
from tfaip.util.multiprocessing.join import Joinable, JoinableHolder

context = multiprocessing.get_context("spawn")
logging = logging.getLogger(__name__)


class ParallelPipeline(Joinable, ABC):
    def __init__(self, holder: JoinableHolder,
                 processes: int,
                 limit: int = -1,
                 auto_repeat_input: bool = False,
                 max_tasks_per_child: int = 250,
                 ):
        super(ParallelPipeline, self).__init__(holder)

        self.limit = limit
        self.auto_repeat_input = auto_repeat_input
        self.processes = processes
        self.max_samples = processes * 32
        self.pool = None
        self.running = False

        if processes > 1:
            # Auto repeat input should only be true during training
            self.pool = WrappedPool(processes=self.processes, worker_constructor=self.create_worker_func(), context=context, maxtasksperchild=max_tasks_per_child)

    def join(self):
        self.running = False
        if self.pool:
            self.pool.close()
            self.pool.join()
            self.pool = None
            
        super(ParallelPipeline, self).join()

    @abstractmethod
    def create_worker_func(self) -> Callable[[], DataWorker]:
        pass

    @abstractmethod
    def generate_input(self):
        pass

    def is_running(self):
        return self.running or (self.pool and self.pool._state == RUN)

    def _max_size_input_generator(self):
        self.running = True
        while self.is_running():
            logging.debug("Input generation start")
            for i, sample in enumerate(self.generate_input()):
                while True:
                    if i == self.limit:
                        # Reached number of desired examples, stop
                        return

                    if not self.is_running():
                        return

                    if self.max_samples > 0:
                        self.max_samples -= 1
                        yield sample
                        break
                    else:
                        time.sleep(0.01)

            if not self.auto_repeat_input:
                break

    def output_generator(self):
        if self.pool is None:
            # Only one thread, run in thread
            worker = self.create_worker_func()()
            worker.initialize_thread()
            for i, o in enumerate(map(worker.process, self._max_size_input_generator())):
                if o is None:
                    logging.debug("Invalid data. Skipping")
                else:
                    yield o
                self.max_samples += 1

        else:
            # Multiprocessing
            for i, o in enumerate(self.pool.imap_gen(self._max_size_input_generator())):
                if o is None:
                    logging.debug("Invalid data. Skipping")
                else:
                    yield o
                self.max_samples += 1
