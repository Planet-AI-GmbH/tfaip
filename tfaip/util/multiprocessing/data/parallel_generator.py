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
import time
from abc import ABC, abstractmethod
from multiprocessing import Value, Lock
from multiprocessing.queues import Queue
from queue import Empty, Full
from typing import Callable, Iterable, Any

from tfaip.util.multiprocessing import context as mp_context
from tfaip.util.multiprocessing.data.worker import DataWorker
from tfaip.util.multiprocessing.sharedmemoryqueue import SharedMemoryQueue, SHARED_MEMORY_SUPPORT
from tfaip.util.logging import logger

logger = logger(__name__)


class Finished:
    pass


def run(worker_fn: Callable[[], DataWorker], max_tasks: int, in_queue: Queue, out_queue: Queue, cancel_process: Value):
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
            try:
                s = in_queue.get(timeout=0.01)
            except Empty:
                logger.debug("In queue empty.")
                if cancel_process.value:
                    logger.debug("Canceling working generator.")
                    break
            else:
                if isinstance(s, Finished):
                    logger.debug("Received Finished. Stopping working generator")
                    break
                yield s

    for out in worker.process(generator()):
        # Process the data and write it to the queue
        while True:
            if cancel_process.value:
                logger.debug("Canceling working processor (inner).")
                break
            try:
                out_queue.put(out, timeout=0.01)
            except Full:
                logger.debug("Out queue Full.")
                continue
            else:
                break

        if cancel_process.value:
            logger.debug("Canceling working processor (outer).")
            break

        max_tasks -= 1
        if max_tasks == 0:
            logger.debug("Max tasks reached for this worker. Stopping.")
            break

    logger.debug("Worker finished")
    if cancel_process.value:
        out_queue.cancel_join_thread()  # this prevents a deadlock if the generator is stopped prematurely


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
        use_shared_memory_queues: bool = SHARED_MEMORY_SUPPORT,
    ):

        self.limit = limit
        self.auto_repeat_input = auto_repeat_input
        self.num_processes = processes
        self.max_in_samples = processes * 32 if max_in_samples < 0 else max_in_samples
        self.max_out_samples = processes * 32 if max_out_samples < 0 else max_out_samples
        self.max_tasks_per_child = max_tasks_per_child
        self.running = False
        self.lock = Lock()
        self.cancel_process = mp_context().Value("i", 0)

        if run_parallel:
            if use_shared_memory_queues and SHARED_MEMORY_SUPPORT:
                self.in_queue = SharedMemoryQueue(maxsize=self.max_in_samples, context=mp_context())
                self.out_queue = SharedMemoryQueue(maxsize=self.max_out_samples, context=mp_context())
            else:
                self._shared_memory_queues = False
                if use_shared_memory_queues:
                    logger.warning(
                        "Shared memory not supported. Python Version >= 3.8 required. Using default queues "
                        "as fallback which might be significantly slower for large numpy arrays."
                    )
                self.in_queue = mp_context().Queue(maxsize=self.max_in_samples)
                self.out_queue = mp_context().Queue(maxsize=self.max_out_samples)
            self.processes = []
            self.input_thread = threading.Thread(target=self._put_inputs, daemon=False)
            self.process_spawner = threading.Thread(target=self._spawn_processes, daemon=False)
        else:
            self.processes = None

    def _spawn_processes(self):
        logger.debug("Started process spawner")

        # Check if subprocesses shall be spawned
        # (e.g. some are not running yet, or some finished to due max_tasks_per_child)
        while not self.cancel_process.value:
            with self.lock:
                if self.processes is None or not self._shall_spawn():
                    break

                self.processes = [p for p in self.processes if p.is_alive()]
                while len(self.processes) < self.num_processes:
                    logger.debug("Spawning new process for generation")
                    self.processes.append(
                        mp_context().Process(
                            target=run,
                            args=(
                                self.create_worker_func(),
                                self.max_tasks_per_child,
                                self.in_queue,
                                self.out_queue,
                                self.cancel_process,
                            ),
                        )
                    )
                    self.processes[-1].start()

            time.sleep(0.1)

        if not self.cancel_process.value:
            # Notify running threads that then can end since no more samples will be written
            for _ in self.processes:
                self.in_queue.put(Finished())

        logger.debug("Stopped process spawner")

    def join(self):
        logger.debug("Attempting to join")
        with self.lock:
            if self.processes:
                logger.debug("Setting cancel_process to stop threads")
                self.cancel_process.value = 1

                for p in self.processes:
                    logger.debug("Joining process")
                    p.join()

                self.processes = None

        logger.debug("Joining process spawner")
        self.process_spawner.join()
        logger.debug("Joining input thread")
        self.input_thread.join()
        logger.debug("all processes joined")

        self.in_queue.close()
        self.out_queue.close()
        self.out_queue.cancel_join_thread()
        self.in_queue.cancel_join_thread()

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
        if self.cancel_process.value:
            return False
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
            while True:
                try:
                    if self.cancel_process.value:
                        logger.debug("Canceling working processor (inner).")
                        break
                    self.in_queue.put(i, timeout=0.01)
                except Full:
                    continue
                else:
                    break

            if self.cancel_process.value:
                logger.debug("Canceling working processor (outer).")
                break

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
            running = False  # store if the data processing is running to check if logging the warning

            # Multiprocessing
            assert self.is_running()
            while self.is_running() or not self.out_queue.empty():
                try:
                    o = self.out_queue.get(timeout=1)
                    running = True  # we got the first example, it is running now.
                except Empty:
                    if running:
                        logger.warning(
                            "Waiting for output from the datapipeline. This means that the training is data-bound."
                            "Try to speed up the data pipeline or use more threads. Or increase the max_tasks_per_process parameter."
                        )
                    continue
                if o is None:
                    logger.debug("Invalid data. Skipping")
                else:
                    yield o
