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
"""Implementation of a TensorflowFix"""
import tensorflow.keras.callbacks as cb


class TensorflowFix(cb.Callback):
    """Fix for a weired Tensorflow bug. Remove this if the Issue is closed or Fixed...

    See https://github.com/tensorflow/tensorflow/issues/42872
    """

    def __init__(self):
        super().__init__()
        self._supports_tf_logs = True  # Any Callback before LAV callback must act on raw tf logs only
        self._backup_loss = None

    def on_train_begin(self, logs=None):
        self._backup_loss = {**self.model.loss}

    def on_train_batch_end(self, batch, logs=None):
        self.model.loss = self._backup_loss
