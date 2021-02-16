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
import tensorflow.keras.callbacks as cb


class FixMetricLabelsCallback(cb.Callback):
    def __init__(self):
        super(FixMetricLabelsCallback, self).__init__()
        self._supports_tf_logs = True
        self.original_metrics = {}

    def on_train_begin(self, logs=None):
        # store original metric names
        self.original_metrics = self.model.compiled_metrics._weighted_metrics

    def on_epoch_end(self, epoch, logs=None):
        self.fix(logs)

    def on_train_batch_end(self, batch, logs=None):
        self.fix(logs)

    def on_predict_batch_end(self, batch, logs=None):
        self.fix(logs)

    def on_test_batch_end(self, batch, logs=None):
        self.fix(logs)

    def fix(self, logs):
        if logs is None:
            return

        for n, m in self.original_metrics.items():
            if not m or not hasattr(m, 'name') or m.name == n:
                continue
            if m.name in logs and n not in logs:
                logs[n] = logs[m.name]
                del logs[m.name]

