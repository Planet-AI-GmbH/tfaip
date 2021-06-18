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
"""Definition of the BenchmarkCallback"""
import logging
from copy import copy
from dataclasses import dataclass

import prettytable
from tensorflow.keras.callbacks import Callback

from tfaip.trainer.callbacks.extract_logs import ExtractLogsCallback
from tfaip.util.profiling import MeasureTime

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResults:
    """Class storing benchmark results of training or testing or prediction

    Use pretty_print to print a formatted table of the full results.
    """

    n_batches: float = 0
    n_samples: float = 0
    total_time: float = 1e-10
    avg_time_per_batch: float = 0
    avg_time_per_sample: float = 0
    batches_per_second: float = 0
    samples_per_second: float = 0

    def pretty_print(self, print_fn=print):
        table = prettytable.PrettyTable(["", "Total", "Batch", "Sample"])
        table.add_row(["Count", 1, self.n_batches, self.n_samples])
        table.add_row(["Time Per", self.total_time, self.avg_time_per_batch, self.avg_time_per_sample])
        table.add_row(["Per Second", 1 / self.total_time, self.batches_per_second, self.samples_per_second])
        if print_fn is None:
            return table
        print_fn(table)

    def finish_epoch(self, duration, samples=None):
        self.total_time = duration
        if samples is not None:
            self.n_samples = samples
        if self.n_batches > 0:
            self.avg_time_per_batch /= self.n_batches
            self.avg_time_per_sample /= self.n_samples
            self.batches_per_second = 1 / self.avg_time_per_batch
            self.samples_per_second = 1 / self.avg_time_per_sample
        return self

    def finish_batch(self, batch_size, batch_time):
        self.n_batches += 1
        if batch_size is not None:
            self.n_samples += batch_size
        self.avg_time_per_batch += batch_time
        self.avg_time_per_sample += batch_time
        return self


class BenchmarkCallback(Callback):
    """
    The BenchmarkCallback will trace the training and validation times per patch, epoch, and in total.
    """

    def __init__(self, extracted_logs_cb: ExtractLogsCallback):
        super().__init__()
        self._supports_tf_logs = True  # Any Callback before LAV callback must act on raw tf logs only
        self.extracted_logs_cb = extracted_logs_cb

        self.total_train_time = MeasureTime()
        self.total_epoch_time = MeasureTime()
        self.total_test_time = MeasureTime()
        self.batch_time = MeasureTime()

        self.last_test_results = BenchmarkResults()
        self.last_train_results = BenchmarkResults()

        self.avg_test_results = BenchmarkResults()
        self.avg_train_results = BenchmarkResults()

    def print(self):
        l_tr = self.last_train_results
        a_tr = copy(self.avg_train_results).finish_epoch(self.total_train_time.duration_till_now())
        l_te = self.last_test_results
        a_te = copy(self.avg_train_results).finish_epoch(self.total_train_time.duration_till_now())
        table = prettytable.PrettyTable(
            ["Benchmark", "Train Total", "Train Batch", "Train Sample", "Test Total", "Test Batch", "Test Sample"]
        )

        def add(prefix, a_tr, a_te):
            table.add_row([prefix + " Count", 1, a_tr.n_batches, a_tr.n_samples, 1, a_te.n_batches, a_te.n_samples])
            table.add_row(
                [
                    prefix + " Time Per",
                    a_tr.total_time,
                    a_tr.avg_time_per_batch,
                    a_tr.avg_time_per_sample,
                    a_te.total_time,
                    a_te.avg_time_per_batch,
                    a_te.avg_time_per_sample,
                ]
            )
            table.add_row(
                [
                    prefix + " Per Second",
                    1 / a_tr.total_time,
                    a_tr.batches_per_second,
                    a_tr.samples_per_second,
                    1 / a_te.total_time,
                    a_te.batches_per_second,
                    a_te.samples_per_second,
                ]
            )

        add("AVG", a_tr, a_te)
        add("Last", l_tr, l_te)
        logger.info("Benchmark results:\n" + str(table))

    def on_train_begin(self, logs=None):
        self.total_train_time.__enter__()

    def on_train_end(self, logs=None):
        self.total_train_time.__exit__(None, None, None)
        self.avg_test_results.finish_epoch(self.total_train_time.duration)
        self.avg_train_results.finish_epoch(self.total_train_time.duration)
        self.print()

    # ========================================================================
    # TRAIN

    def on_epoch_begin(self, epoch, logs=None):
        self.total_epoch_time.__enter__()
        self.last_train_results = BenchmarkResults()

    def on_epoch_end(self, epoch, logs=None):
        self.total_epoch_time.__exit__(None, None, None)
        count = self.extracted_logs_cb.shadow_logs["count"]
        self.last_train_results.finish_epoch(self.total_epoch_time.duration, count)
        self.avg_train_results.n_samples += count
        self.print()

    def on_train_batch_begin(self, batch, logs=None):
        self.batch_time.__enter__()

    def on_train_batch_end(self, batch, logs=None):
        self.batch_time.__exit__(None, None, None)
        self.last_train_results.finish_batch(None, self.batch_time.duration)
        self.avg_train_results.finish_batch(None, self.batch_time.duration)

    # ===================================================================
    # TEST

    def on_test_begin(self, logs=None):
        self.total_test_time.__enter__()
        self.last_test_results = BenchmarkResults()

    def on_test_end(self, logs=None):
        count = self.extracted_logs_cb.shadow_logs["count"]
        self.total_test_time.__exit__(None, None, None)
        self.last_test_results.finish_epoch(self.total_test_time.duration, count)
        self.avg_test_results.total_time += self.total_test_time.duration
        self.avg_test_results.n_samples += count

    def on_test_batch_begin(self, batch, logs=None):
        self.batch_time.__enter__()

    def on_test_batch_end(self, batch, logs=None):
        self.batch_time.__exit__(None, None, None)
        self.last_test_results.finish_batch(None, self.batch_time.duration)
        self.avg_test_results.finish_batch(None, self.batch_time.duration)
