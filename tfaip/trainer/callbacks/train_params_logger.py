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
"""Definition of the TrainParamsLogger"""
import json
import os
import sys

import tensorflow.keras as keras
import logging

from tfaip.util.tftyping import sync_to_numpy_or_python_type


logger = logging.getLogger(__name__)


class TrainerCheckpointsCallback(keras.callbacks.ModelCheckpoint):
    """
    Callback to store the current state of the trainer params and the current training model with
    all of its weights which is required for resuming the training.

    This is realized by reimplementing some of the methods of the keras ModelCheckpoint-Callback which is a base class.
    """

    def __init__(self, train_params, save_freq=None, store_weights=True, store_params=True):
        # before the actual ModelCheckpoint base is initialized, determine the ckpt_dir
        log_dir = train_params.output_dir
        if log_dir and train_params.checkpoint_sub_dir:
            log_dir = os.path.join(log_dir, train_params.checkpoint_sub_dir)

        ckpt_dir = os.path.join(log_dir, "variables", "variables") if log_dir else ""

        if ckpt_dir is None or save_freq is None:
            save_freq = sys.maxsize  # never

        # Init of the parent
        super().__init__(
            ckpt_dir,
            save_weights_only=True,
            save_freq=save_freq,
        )

        # Set additional members
        self.log_dir = log_dir
        self.train_params = train_params
        self.store_weights = store_weights
        self.store_params = store_params

    def _save_model(self, epoch, logs):
        # Override save model to either store weights or the params
        if self.store_params:
            if isinstance(self.save_freq, int) or self.epochs_since_last_save >= self.period:
                logs = sync_to_numpy_or_python_type(logs)
                # Crop variables/variables of the path
                filepath = os.path.abspath(os.path.join(self._get_file_path(epoch, logs), "..", ".."))
                self.train_params.current_epoch = epoch + 1
                self.train_params.saved_checkpoint_sub_dir = os.path.relpath(filepath, self.train_params.output_dir)
                self._save_params(filepath)

        if self.store_weights:
            super()._save_model(epoch, logs)

    def _save_params(self, filepath):
        # Save the params only
        os.makedirs(filepath, exist_ok=True)
        path = os.path.join(filepath, "trainer_params.json")
        logger.debug(f"Logging current trainer state to '{path}'")
        with open(path, "w") as f:
            json.dump(self.train_params.to_dict(), f, indent=2)

    def on_epoch_end(self, epoch, logs=None):
        # we ended an epoch, store that we ended it
        self.train_params.current_epoch = epoch + 1
        super().on_epoch_end(epoch, logs)
