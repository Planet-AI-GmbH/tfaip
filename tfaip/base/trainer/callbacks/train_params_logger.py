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
import sys

import tensorflow.keras as keras
from tensorflow.python.keras.utils import tf_utils
import logging


logger = logging.getLogger(__name__)


class TrainParamsLoggerCallback(keras.callbacks.ModelCheckpoint):
    def __init__(self, train_params, save_freq=None):
        log_dir = train_params.checkpoint_dir
        if log_dir and train_params.checkpoint_sub_dir_:
            log_dir = os.path.join(log_dir, train_params.checkpoint_sub_dir_)

        ckpt_dir = os.path.join(log_dir, 'variables', 'variables') if log_dir else ''

        if ckpt_dir is None or save_freq is None:
            save_freq = sys.maxsize  # never

        super(TrainParamsLoggerCallback, self).__init__(
            ckpt_dir,
            save_weights_only=True,
            save_freq=save_freq,
        )
        self.log_dir = log_dir
        self.train_params = train_params

    def _save_model(self, epoch, logs):
        if isinstance(self.save_freq,
                      int) or self.epochs_since_last_save >= self.period:
            logs = tf_utils.to_numpy_or_python_type(logs)
            filepath = os.path.abspath(os.path.join(self._get_file_path(epoch, logs), '..', '..'))  # no variables/variables
            self.train_params.current_epoch = epoch + 1
            self.train_params.saved_checkpoint_sub_dir_ = os.path.relpath(filepath, self.train_params.checkpoint_dir)
            self._save_params(filepath)

        super(TrainParamsLoggerCallback, self)._save_model(epoch, logs)

    def _save_params(self, filepath):
        os.makedirs(filepath, exist_ok=True)
        path = os.path.join(filepath, 'trainer_params.json')
        logger.debug("Logging current trainer state to '{}'".format(path))
        with open(path, 'w') as f:
            json.dump(self.train_params.to_dict(), f, indent=2)

    def on_epoch_end(self, epoch, logs=None):
        # we ended an epoch, store that we ended it
        self.train_params.current_epoch = epoch + 1
        super(TrainParamsLoggerCallback, self).on_epoch_end(epoch, logs)
