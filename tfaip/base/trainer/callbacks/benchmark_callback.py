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
from tensorflow.keras.callbacks import Callback
from time import time, strftime, gmtime
import logging

logger = logging.getLogger(__name__)


class RunningAverage:
    def __init__(self):
        self.steps = 0
        self.avg = 0

    def add(self, v):
        self.avg = (self.avg * self.steps + v) / (self.steps + 1)
        self.steps += 1

    def reset(self):
        self.steps = 0
        self.avg = 0


class BenchmarkCallback(Callback):
    def __init__(self):
        super(BenchmarkCallback, self).__init__()
        self._supports_tf_logs = True       # Any Callback before LAV callback must act on raw tf logs only
        self.train_start_time = -1
        self.epoch_start_time = -1
        self.batch_start_time = -1
        self.test_start_time = -1

        self.avg_time_per_batch = RunningAverage()
        self.avg_time_per_epoch = RunningAverage()
        self.avg_time_per_test = RunningAverage()

        self.last_dt_per_epoch = -1
        self.last_dt_per_test = -1
        self.last_dt_per_batch = RunningAverage()

    def print(self):
        def fmt(s):
            return strftime('%M:%S', gmtime(int(s))) + '.' + '{:.2f}'.format(s - int(s))[2:]
        logger.info("Timing TOTAL {} | EPOCH {} | TRAIN {} | TEST {} | BATCH {:.3f}s".format(
            strftime('%H:%M:%S', gmtime(time() - self.train_start_time)),
            fmt(self.avg_time_per_epoch.avg),
            fmt(self.avg_time_per_epoch.avg - self.avg_time_per_test.avg),
            fmt(self.avg_time_per_test.avg),
            self.avg_time_per_batch.avg
        ))
        logger.info("Timing LAST           | EPOCH {} | TRAIN {} | TEST {} | BATCH {:.3f}s".format(
            fmt(self.last_dt_per_epoch),
            fmt(self.last_dt_per_epoch - self.last_dt_per_test),
            fmt(self.last_dt_per_test),
            self.last_dt_per_batch.avg,
            ))

    def on_train_begin(self, logs=None):
        self.train_start_time = time()

    def on_train_end(self, logs=None):
        self.print()

    def on_epoch_begin(self, epoch, logs=None):
        self.last_dt_per_batch.reset()
        self.epoch_start_time = time()

    def on_epoch_end(self, epoch, logs=None):
        self.last_dt_per_epoch = time() - self.epoch_start_time
        self.avg_time_per_epoch.add(self.last_dt_per_epoch)
        self.print()

    def on_train_batch_begin(self, batch, logs=None):
        self.batch_start_time = time()

    def on_train_batch_end(self, batch, logs=None):
        dt_batch = time() - self.batch_start_time
        self.last_dt_per_batch.add(dt_batch)
        self.avg_time_per_batch.add(dt_batch)

    def on_test_begin(self, logs=None):
        self.test_start_time = time()

    def on_test_end(self, logs=None):
        self.last_dt_per_test = time() - self.test_start_time
        self.avg_time_per_test.add(self.last_dt_per_test)
