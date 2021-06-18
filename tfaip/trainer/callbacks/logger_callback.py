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
"""Definition of the LoggerCallback"""
from tensorflow.keras.callbacks import Callback
import logging
import numpy as np


logger = logging.getLogger(__name__)


class LoggerCallback(Callback):
    """
    The logger callback prints useful information about the training process:
    - log at the end of a epoch a
    - Write the current epoch
    - Store the logs of the previous epoch


    This is required for the train.log where the progress bar and thus the metrics are not written to.
    """

    def __init__(self):
        super().__init__()
        self.last_logs = {}

    def on_epoch_begin(self, epoch, logs=None):
        self.last_logs = logs
        logger.info(f"Start of epoch {epoch + 1:4d}")

    def on_epoch_end(self, epoch, logs=None):
        self.last_logs = logs
        if logs is None:
            return
        logs_str = " - ".join(f"{k}: {np.mean(logs[k]):.4f}" for k in sorted(logs.keys()))
        logger.info(f"Results of epoch {epoch + 1:4d} {logs_str}")
