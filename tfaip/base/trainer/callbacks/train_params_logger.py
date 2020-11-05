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
import json
import os

import tensorflow.keras as keras
import logging


logger = logging.getLogger(__name__)


class TrainParamsLoggerCallback(keras.callbacks.Callback):
    def __init__(self, train_params, log_dir=None):
        super(TrainParamsLoggerCallback, self).__init__()
        self.log_dir = log_dir
        self.train_params = train_params

    def _save(self):
        if self.log_dir:
            path = os.path.join(self.log_dir, 'trainer_params.json')
            logger.debug("Logging current trainer state to '{}'".format(path))
            with open(path, 'w') as f:
                json.dump(self.train_params.to_dict(), f, indent=2)

    def on_train_begin(self, logs=None):
        self._save()

    def on_epoch_end(self, epoch, logs=None):
        # we ended an epoch, store that we ended it
        self.train_params.current_epoch = epoch + 1
        self._save()

