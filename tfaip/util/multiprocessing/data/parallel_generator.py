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
"""Definition of a multiprocessing Pool for running generators in parallel"""
import threading
from abc import ABC, abstractmethod
import multiprocessing
from multiprocessing.queues import Queue
import time
from queue import Empty
from typing import Callable, Iterable, Any
import logging

from tfaip.util.multiprocessing import context as mp_context
from tfaip.util.multiprocessing.data.worker import DataWorker
from tfaip.util.multiprocessing.join import Joinable, JoinableHolder

logger = logging.getLogger(__name__)


class Finished:
    pass


def run(worker_fn: Callable[[], DataWorker], max_tasks: int, in_queue: Queue, out_queue: Queue):
    # This is the worker function of a spawned process.
    # While the task does not receive a Finished flag, if fetches an input sample which is yielded in the worker.
    # The outputs are then written to the output queue.
    # Each thread is stopping if max_tasks is reached.
    logger.debug("Worker starting")
    worker = worker_fn()
    worker.initialize_thread()

    def generator():
        # Transform the input queue to a generator
        # Note that other processes read from the sample in_queue
        while True:
            s = in_queue.get()
            if isinstance(s, Finished):
                logger.debug("Received None. Stopping worker")
                break
            yield s

    for out in worker.process(generator()):
        # Process the data and write it to the queue
        out_queue.put(out)
        max_tasks -= 1
        if max_tasks == 0:
            logger.debug("Max tasks reached for this worker. Stopping.")
            break

    logger.debug("Worker finished")


class ParallelGenerator(ABC):
    """
    Basic implementation to parallelize generators.

    Assume you have n inputs that are processed by i generators yielding m outputs in total.
    This implementation writes all inputs to a queue, i generators are spawned, each gets a sample and writes its output
    to an output queue. The data of the output queue is yielded.
    Note that the data in the output queue is not sorted since the generator outputs are interleaved.

    When using the ParallelGenerator, implement the worker and generation functions.
    The ParallelGenerator must be called in a `with`-block to ensure that the processes are launched and joined.

    The implementation is oriented at the implementation of multiprocessing.Pool
    Likewise, a max_tasks_per_child parameter defines how many tasks a process can process before stopped and a new
    one is launched. This is required since sometimes the memory of sup-processes is not freed automatically.
    """

    def __init__(
        self,
        processes: int,
        limit: int = -1,
        auto_repeat_input: bool = False,
        max_tasks_per_child: int = 250,
        run_parallel: bool = True,  # Flag to warning and disable parallel run
        max_in_samples: int = -1,
        max_out_samples: int = -1,
    ):

        self.limit = limit
        self.auto_repeat_input = auto_repeat_input
        self.num_processes = processes
        self.max_in_samples = processes * 32 if max_in_samples < 0 else max_in_samples
        self.max_out_samples = processes * 32 if max_out_samples < 0 else max_out_samples
        self.max_tasks_per_child = max_tasks_per_child
        self.running = False

        if run_parallel:
            self.in_queue = mp_context().Queue(maxsize=self.max_in_samples)
            self.out_queue = mp_context().Queue(maxsize=self.max_out_samples)
            self.processes = []
            self.input_thread = threading.Thread(target=self._put_inputs, daemon=True)
            self.process_spawner = threading.Thread(target=self._spawn_processes, daemon=True)
        else:
            self.processes = None

    def _spawn_processes(self):
        logger.debug("Started process spawner")

        # Check if subprocesses shall be spawned
        # (e.g. some are not running yet, or some finished to due max_tasks_per_child)
        while self._shall_spawn():
            self.processes = [p for p in self.processes if p.is_alive()]
            while len(self.processes) < self.num_processes:
                logger.debug("Spawning new process for generation")
                self.processes.append(
                    mp_context().Process(
                        target=run,
                        args=(self.create_worker_func(), self.max_tasks_per_child, self.in_queue, self.out_queue),
                    )
                )
                self.processes[-1].start()

            time.sleep(0.1)
        # stop the remaining processes by sending a Finished signal
        for _ in self.processes:
            logger.debug("Putting Finished() to stop remaining threads")
            self.in_queue.put(Finished())

        # Wait for finished
        for p in self.processes:
            p.join()

        # remove additional Finished (can happen if max_tasks_per_child was reached before reading Finished)
        while not self.in_queue.empty():
            v = self.in_queue.get_nowait()
            assert isinstance(v, Finished)

        logger.debug("Stopped process spawner")

    def join(self):
        logger.debug("Attempting to join")
        if self.processes:
            for p in self.processes:
                p.join()
            self.in_queue.close()
            self.out_queue.close()
            self.processes = None
        logger.debug("ParallelGenerator joined")

    def __enter__(self) -> Iterable[Any]:
        self.input_thread.start()
        while not self.is_running():
            time.sleep(0.01)
        return self._output_generator()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.join()

    @abstractmethod
    def create_worker_func(self) -> Callable[[], DataWorker]:
        pass

    @abstractmethod
    def generate_input(self) -> Iterable[Any]:
        """
        Implementation to Generate inputs
        """
        pass

    def is_running(self):
        return any(p.is_alive() for p in self.processes) or self._shall_spawn() or not self.out_queue.empty()

    def _shall_spawn(self):
        return self.input_thread.is_alive() or not self.in_queue.empty()

    def _max_size_input_generator(self):
        # wrapping of the input generator function (`generate_input`) to support limit and auto-repeat.
        logger.debug("Input generation start")
        while self.is_running():
            logger.debug("Input generation epoch start")
            for i, sample in enumerate(self.generate_input()):
                if i == self.limit:
                    # Reached number of desired examples, stop
                    return

                if not self.is_running():
                    return

                yield sample

            if not self.auto_repeat_input:
                break

        logger.debug("Input generation stop")

    def _put_inputs(self):
        logger.debug("Input thread started")
        self.process_spawner.start()  # start the process spawner when input runner was started

        for i in self._max_size_input_generator():
            self.in_queue.put(i)

        logger.debug("Input thread finished")

    def _output_generator(self) -> Iterable[Any]:
        """
        Retrieve the outputs
        """
        if self.processes is None:
            # Only one thread, run in thread
            worker = self.create_worker_func()()
            worker.initialize_thread()
            for o in self._max_size_input_generator():
                for oo in worker.process(o):
                    if oo is None:
                        logger.debug("Invalid data. Skipping")
                    else:
                        yield oo
        else:
            # Multiprocessing
            assert self.is_running()
            while self.is_running() or not self.out_queue.empty():
                try:
                    o = self.out_queue.get(timeout=1)
                except Empty:
                    continue
                if o is None:
                    logger.debug("Invalid data. Skipping")
                else:
                    yield o
