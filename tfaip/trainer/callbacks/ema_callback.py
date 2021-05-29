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
"""Definition of the EMACallback"""
from tensorflow.keras.callbacks import Callback
import logging

from typeguard import typechecked

from tfaip.trainer.optimizer.weights_moving_average import WeightsMovingAverage


logger = logging.getLogger(__name__)


class EMACallback(Callback):
    """The EMACallback swaps the weights of the model with EMA or non EMA which is required for validation and export.

    For example, at the begin of testing the EMA weights are loaded, and at the end the original weigs are restored.
    Similarly, at the end of a epoch the EMA weights are loaded to export the prediction model, and at the end of
    each epoch the weights are reset to the actual weights.
    """

    @typechecked
    def __init__(self, optimizer: WeightsMovingAverage):
        # any callback after this one will have ema weights on epoch end
        # (useful for exporting best, but not checkpointing)
        super().__init__()
        self.optimizer = optimizer
        self._supports_tf_logs = True

    def on_test_begin(self, logs=None):
        self.optimizer.to_avg(self.model.variables)

    def on_test_end(self, logs=None):
        self.optimizer.to_model(self.model.variables)

    def on_epoch_begin(self, epoch, logs=None):
        self.optimizer.to_model(self.model.variables)

    def on_epoch_end(self, epoch, logs=None):
        self.optimizer.to_avg(self.model.variables)
