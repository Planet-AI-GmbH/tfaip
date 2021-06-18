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
"""Definition of the ExtractLogsCallback"""
from typing import Optional

import tensorflow.keras.callbacks as cb
from tfaip.util.tftyping import sync_to_numpy_or_python_type

from tfaip.trainer.callbacks.tensor_board_data_handler import TensorBoardDataHandler


class ExtractLogsCallback(cb.Callback):
    """This callback is a utility to extract variables from the log that can not be handled by all callbacks.

    The actual use-case is to log custom data to the TensorBoard (e.g., bytes or images)
    The values will be added to the logs since they are "Metrics" (this is a bit hacky...), however they must
    immediately removed from the logs as they are "not real logs" to be displayed, but only used by the
    TensorBoardCallback.

    Thereto, all logs are extracted and stored in a separated data structure which is cleared on the begin of the
    training. The TensorBoardCallback has then access to the extracted logs.

    The ``extracted_logs`` will be passed to the tensorboard handler.
    The ``shadow_logs`` will only be stored. They are for example used to store the "counts" of batches which are
    required by the BenchmarkCallback.
    """

    def __init__(self, test_prefix="val_"):
        super().__init__()
        self._supports_tf_logs = True
        self.tensorboard_data_handler: Optional[TensorBoardDataHandler] = None
        self.extracted_logs = {}
        self.shadow_logs = {}
        self.test_prefix = test_prefix

    def set_model(self, model):
        super().set_model(model)
        self.tensorboard_data_handler = TensorBoardDataHandler(self.model)

    def on_train_begin(self, logs=None):
        self.extracted_logs = {}
        self.shadow_logs = {}

    def on_epoch_begin(self, epoch, logs=None):
        self.extracted_logs = {}
        self.shadow_logs = {}

    def on_epoch_end(self, epoch, logs=None):
        self.extract(logs)

    def on_train_batch_end(self, batch, logs=None):
        self.extract(logs)

    def on_predict_batch_end(self, batch, logs=None):
        self.extract(logs)

    def on_test_batch_end(self, batch, logs=None):
        # extract the logs, but add val_ as prefix (as common)
        self.extract(logs, prefix=self.test_prefix)

    def on_test_end(self, logs=None):
        # extract the logs, but add val_ as prefix (as common)
        self.extract(logs, prefix=self.test_prefix)

    def extract(self, logs, prefix=""):
        if logs is None:
            return
        if "count" in logs:
            self.shadow_logs["count"] = sync_to_numpy_or_python_type(logs["count"])
            del logs["count"]
        for k in list(logs.keys()):
            if self.tensorboard_data_handler.is_tensorboard_only(k, logs[k]):
                self.extracted_logs[prefix + k] = logs[k].numpy() if hasattr(logs[k], "numpy") else logs[k]
                del logs[k]
