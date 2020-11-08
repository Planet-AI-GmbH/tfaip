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
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from dataclasses_json import dataclass_json
from tensorflow.keras.callbacks import Callback

if TYPE_CHECKING:
    from tfaip.base.scenario.scenariobase import ScenarioBase
    from tfaip.base.trainer.trainer import TrainerParams

logger = logging.getLogger(__name__)


@dataclass_json
@dataclass
class EarlyStoppingParams:
    # Logging parameters -> set from ModelBase
    mode_: str = None
    current_: float = None  # If loaded from a checkpoint, this value is already set
    monitor_: str = None
    n_: int = -1

    # User parameters
    # TODO: help
    best_model_output_dir: Optional[str] = None
    best_model_name: str = 'best'
    frequency: int = 1
    n_to_go: int = -1

    # Thresholds: either ignore in early stopping, or stop immediately
    # e.g. 0 and 1 for accuracy
    lower_threshold: float = -1e100
    upper_threshold: float = 1e100


class EarlyStoppingCallback(Callback):
    def __init__(self, scenario: 'ScenarioBase', trainer_params: 'TrainerParams'):
        # mode = min or max
        super(EarlyStoppingCallback, self).__init__()
        self._trainer_params = trainer_params
        self._params = trainer_params.early_stopping_params
        if self._params.best_model_output_dir is None:
            self._params.best_model_output_dir = trainer_params.checkpoint_dir

        self._export_best = trainer_params.export_best and self._params.best_model_output_dir is not None
        self._scenario = scenario
        self._export_dir = os.path.join(self._params.best_model_output_dir, self._params.best_model_name) if self._export_best else None

        if self._params.monitor_ is None or self._params.mode_ is None:
            mode, monitor = scenario.best_logging_settings()
            mode = mode.lower()
            if not monitor.startswith("val_"):
                logger.info(f"Automatically selecting 'val_{monitor}' instead of '{monitor}' for monitoring.")
                monitor = "val_" + monitor

            assert mode in ["min", "max"]
            self._params.monitor_, self._params.mode_ = monitor, mode
        else:
            # Already defined from loaded checkpoint (e.g. resume training)
            pass

        self._early_stopping_enabled = self._params.n_to_go > 1 and self._params.frequency > 0
        if self._early_stopping_enabled:
            logger.info(f"Early stopping enabled with n_to_go={self._params.n_to_go} and frequency={self._params.frequency}.")
        else:
            logger.info("Early stopping deactivated.")

    def on_epoch_end(self, epoch, logs=None):
        if self._params.monitor_ not in logs:
            if self._params.monitor_.startswith("lav_"):
                logger.debug(
                    f"{self._monitor} was not found in logs. Maybe this epoch lav was not called. Skipping export best")
                return

            allowed_suffixes = ['_loss', '_metric']
            initial = self._params.monitor_
            for suf in allowed_suffixes:
                if self._params.monitor_ + suf in logs:
                    self._params.monitor_ += suf
                    break

            if initial != self._params.monitor_:
                logger.warning("{} was not found in logs, using {} instead.".format(initial, self._params.monitor_))
            else:
                logger.error("Could not find '{}' in logs: {}".format(self._params.monitor_, logs))
                return

        new_value = logs.get(self._params.monitor_)
        better_found = False
        if self._params.current_ is None:
            better_found = True
        elif self._params.mode_ == 'min':
            if new_value < self._params.current_:
                better_found = True
        else:
            if new_value > self._params.current_:
                better_found = True

        if better_found:
            logger.info(f"Better value of {self._params.monitor_} found. Old = {self._params.current_}, Best = {new_value}")
            self._params.current_ = new_value
            if self._export_best:
                self._scenario.export(self._export_dir, self._trainer_params)
            self._params.n_ = 1
            if self._early_stopping_enabled:
                logger.debug(f"Early stopping reset. Iteration to go = {self._params.n_to_go}")
        else:
            if self._early_stopping_enabled:
                if (epoch + 1) % self._params.frequency == 0:
                    if self._params.mode_ == 'max' and self._params.current_ < self._params.lower_threshold:
                        logger.info(f"Not counting {self._params.current_} for early stopping since lower threshold {self._params.lower_threshold} was not reached")
                    elif self._params.mode_ == 'min' and self._params.current_ > self._params.upper_threshold:
                        logger.debug(f"Not counting {self._params.current_} for early stopping since upper threshold {self._params.upper_threshold} was not reached")
                    else:
                        logger.info(f"Early stopping progressed. (remaining iteration without improvement: {self._params.n_to_go - self._params.n_})")
                        self._params.n_ += 1
                else:
                    logger.debug(f"Epoch {epoch + 1} ignored in early stopping as frequency={self._params.frequency}.")

            logger.info(f"No better value of {self._params.monitor_} found. Worse = {new_value}, Best = {self._params.current_}")

        if self._params.mode_ == 'min':
            if self._params.current_ <= self._params.lower_threshold:
                self.model.stop_training = True
                logger.info(f"Early stopping. Reached limit value of {self._params.monitor_}: {self._params.current_} <= {self._params.lower_threshold}")
        else:
            if self._params.current_ >= self._params.upper_threshold:
                self.model.stop_training = True
                logger.info(f"Early stopping. Reached limit value of {self._params.monitor_}: {self._params.current_} >= {self._params.upper_threshold}")

        if self._early_stopping_enabled and self._params.n_ >= self._params.n_to_go:
            self.model.stop_training = True
            logger.info(f"Early stopping. Reached number of maximum iterations without improvement ({self._params.n_} = {self._params.n_to_go}")
