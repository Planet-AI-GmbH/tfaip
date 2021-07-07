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
import logging
from contextlib import ExitStack
from queue import Queue
from threading import Thread
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tfaip.predict.predictorbase import PredictorBase

logger = logging.getLogger(__name__)


class StopSignal:
    ...


class RawPredictorThread(Thread):
    def __init__(self, in_queue: Queue, out_queue: Queue, predictor: "PredictorBase"):
        super().__init__(daemon=True)
        self.predictor = predictor
        self.in_queue = in_queue
        self.out_queue = out_queue

    def run(self) -> None:
        def generator():
            while True:
                sample = self.in_queue.get()
                if isinstance(sample, StopSignal):
                    break
                yield sample

        for sample in self.predictor.predict_raw(generator(), size=5):
            self.out_queue.put(sample)


class RawPredictorCaller:
    def __init__(self, in_queue, out_queue):
        self.in_queue = in_queue
        self.out_queue = out_queue

    def __call__(self, sample):
        self.in_queue.put(sample)
        return self.out_queue.get()


class RawPredictor:
    """Utility class to allow for raw prediction but keeping the internal queues open."""

    def __init__(self, predictor: "PredictorBase"):
        self.predictor = predictor
        if self.predictor.params.pipeline.batch_size != 1:
            logger.warning(f"Raw prediction via threading requires batch size == 1. Automatically setting.")
            self.predictor.params.pipeline.batch_size = 1
        if not self.predictor.params.silent:
            logger.warning(f"Consider setting predictor to silent by setting predictor_params.silent = True.")

        self.exit_stack = ExitStack()
        self.in_queue = Queue(10)
        self.out_queue = Queue(10)
        self.thread = RawPredictorThread(self.in_queue, self.out_queue, predictor)

    def __enter__(self):
        self.thread.start()
        return RawPredictorCaller(self.in_queue, self.out_queue)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.in_queue.put(StopSignal())
        self.thread.join()
