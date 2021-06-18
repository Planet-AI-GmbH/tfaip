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
"""Definition of the TensorBoardCallback"""
import glob
import logging
import os
from typing import TYPE_CHECKING

import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.ops import summary_ops_v2

if TYPE_CHECKING:
    from tfaip.trainer.callbacks.extract_logs import ExtractLogsCallback

logger = logging.getLogger(__name__)


class TensorBoardCallback(TensorBoard):
    """Custom implementation fo the TensorBoard-Callback of keras to provide additional functionality:

    - Logging of LAV (custom LAV writer)
    - Adding of the learning rate
    - Applying a TensorBoardDataHandler on every log output to handle custom tensorboard data (e.g. PR-Curves or images)
    """

    def __init__(
        self, log_dir, steps_per_epoch, extracted_logs_cb: "ExtractLogsCallback", reset=False, profile=0, **kwargs
    ):
        if reset:
            logger.info(f'Removing old event files from "{log_dir}"')
            for f in glob.glob(os.path.join(log_dir, "*", "events.out.tfevents*")):
                logger.debug(f"Removing old event '{f}'")
                os.remove(f)

        super().__init__(log_dir=log_dir, profile_batch=profile, **kwargs)
        self.steps_per_epoch = steps_per_epoch
        self.extracted_logs_cb = extracted_logs_cb

        self._lav_dir = os.path.join(log_dir, "lav")

    def _lav_writer(self, idx):
        label = f"lav_l{idx}"
        if label not in self._writers:
            lav_dir = f"{self._lav_dir}_l{idx}"
            os.makedirs(lav_dir, exist_ok=True)
            self._writers[label] = summary_ops_v2.create_file_writer_v2(lav_dir)
        return self._writers[label]

    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        super().on_epoch_begin(epoch, logs)

    def _log_epoch_metrics(self, epoch, logs):
        # custom override, to enable lav logging and applying the TensorBoardDataHandler
        if not logs:
            return

        logs = logs.copy()

        logs.update({"lr": K.eval(self.model.optimizer.lr(epoch * self.steps_per_epoch))})
        if self.extracted_logs_cb:
            logs.update(self.extracted_logs_cb.extracted_logs)

        train_logs = {k: v for k, v in logs.items() if not k.startswith("val_") and not k.startswith("lav_")}
        val_logs = {k: v for k, v in logs.items() if k.startswith("val_")}
        lav_logs = {k: v for k, v in logs.items() if k.startswith("lav_")}

        with summary_ops_v2.always_record_summaries():  # pylint: disable=not-context-manager
            if train_logs:
                with self._train_writer.as_default():
                    for name, value in train_logs.items():
                        self.extracted_logs_cb.tensorboard_data_handler.handle(name, "epoch_" + name, value, step=epoch)
            if val_logs:
                with self._val_writer.as_default():
                    for name, value in val_logs.items():
                        name = name[4:]  # Remove 'val_' prefix.
                        self.extracted_logs_cb.tensorboard_data_handler.handle(name, "epoch_" + name, value, step=epoch)

            # lav logs, include lav list idx
            if lav_logs:
                lavs = {}
                for k, v in lav_logs.items():
                    k = k[4:]  # Remove 'lav_' prefix
                    k = k[1:]  # Remove 'l' prefix
                    d = k.find("_")
                    lav_idx = k[:d]
                    k = k[d + 1 :]  # Remove 'digit + _'
                    if lav_idx not in lavs:
                        lavs[lav_idx] = {}
                    lavs[lav_idx][k] = v

                for idx, v in lavs.items():
                    with self._lav_writer(idx).as_default():
                        for name, value in v.items():
                            self.extracted_logs_cb.tensorboard_data_handler.handle(
                                name, "epoch_" + name, value, step=epoch
                            )
