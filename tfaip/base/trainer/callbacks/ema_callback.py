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
import logging

from typeguard import typechecked

from tfaip.base.trainer.optimizer.weights_moving_average import WeightsMovingAverage


logger = logging.getLogger(__name__)


class EMACallback(Callback):
    @typechecked
    def __init__(self, optimizer: WeightsMovingAverage):
        # any callback after this one will have ema weights on epoch end
        # (useful for exporting best, but not checkpointing)
        super(EMACallback, self).__init__()
        self.optimizer = optimizer

    def on_test_begin(self, logs=None):
        self.optimizer.to_avg(self.model.variables)

    def on_test_end(self, logs=None):
        self.optimizer.to_model(self.model.variables)

    def on_epoch_begin(self, epoch, logs=None):
        self.optimizer.to_model(self.model.variables)

    def on_epoch_end(self, epoch, logs=None):
        self.optimizer.to_avg(self.model.variables)
