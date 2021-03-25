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
"""Definition of the FixLogLabelsCallback"""
import tensorflow.keras.callbacks as cb


class FixLogLabelsCallback(cb.Callback):
    """
    By default tensorflow labels the metrics (metric.name) by a functions name even though they were correctly named
    as metric.
    The same holds for the names of the losses.
    This callback stores the original correct names, and renames the keys of the logs to be correct by calling fix.
    """

    def __init__(self):
        super().__init__()
        self._supports_tf_logs = True
        self.original_metrics = {}  # store the complete dict of original metrics
        self.loss_names = []  # store the losses
        self.initialized = False

    def on_train_begin(self, logs=None):
        # This is a bit hacky, but this is the only time to access the actual metric an loss names...
        if not self.initialized:
            self.original_metrics = {
                **self.model.compiled_metrics._weighted_metrics,  # pylint: disable=protected-access
                **self.model.compiled_metrics._metrics,  # pylint: disable=protected-access
            }
            self.loss_names = list(self.model.compiled_loss._losses.keys())  # pylint: disable=protected-access
            self.initialized = True

    def on_epoch_end(self, epoch, logs=None):
        self.fix(logs)

    def on_train_batch_end(self, batch, logs=None):
        self.fix(logs)

    def on_predict_batch_end(self, batch, logs=None):
        self.fix(logs)

    def on_test_batch_end(self, batch, logs=None):
        self.fix(logs)

    def fix(self, logs: dict):
        if logs is None:
            return

        for name in list(logs.keys()):
            if 'multi_metric' in name:
                del logs[name]

        # first loss metric is the averaged (which is just named 'loss')
        assert len(self.loss_names) == len(self.model.compiled_loss.metrics) - 1
        for name, loss_metric in zip(self.loss_names, self.model.compiled_loss.metrics[1:]):
            if loss_metric.name in logs and name not in logs:
                logs[name] = logs[loss_metric.name]
                del logs[loss_metric.name]

        # Update metric names
        for n, m in self.original_metrics.items():
            if not m or not hasattr(m, 'name') or m.name == n:
                continue
            if m.name in logs and n not in logs:
                logs[n] = logs[m.name]
                del logs[m.name]
