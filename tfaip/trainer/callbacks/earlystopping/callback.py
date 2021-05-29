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
"""Definition of the EarlyStoppingCallback"""
import logging
import os
from typing import TYPE_CHECKING

from tensorflow.keras.callbacks import Callback

if TYPE_CHECKING:
    from tfaip.scenario.scenariobase import ScenarioBase
    from tfaip.trainer.params import TrainerParams

logger = logging.getLogger(__name__)


class EarlyStoppingCallback(Callback):
    """
    Callback that implements early stopping and also (always) tracks the best model.
    """

    def __init__(self, scenario: "ScenarioBase", trainer_params: "TrainerParams"):
        super().__init__()
        self._trainer_params = trainer_params
        self._params = trainer_params.early_stopping
        if self._params.best_model_output_dir is None:
            self._params.best_model_output_dir = trainer_params.output_dir

        self._export_best = trainer_params.export_best and self._params.best_model_output_dir is not None
        self._scenario = scenario
        self._export_dir = (
            os.path.join(self._params.best_model_output_dir, self._params.best_model_name)
            if self._export_best
            else None
        )

        if self._params.monitor is None or self._params.mode is None:
            mode, monitor = scenario.best_logging_settings()
            mode = mode.lower()
            if not monitor.startswith("val_"):
                logger.info(f"Automatically selecting 'val_{monitor}' instead of '{monitor}' for monitoring.")
                monitor = "val_" + monitor

            assert mode in ["min", "max"]
            self._params.monitor, self._params.mode = monitor, mode
        else:
            # Already defined from loaded checkpoint (e.g. resume training)
            pass

        self._early_stopping_enabled = self._params.n_to_go > 1 and self._params.frequency > 0
        if self._early_stopping_enabled:
            logger.info(
                f"Early stopping enabled with n_to_go={self._params.n_to_go} and frequency={self._params.frequency}."
            )
        else:
            logger.info("Early stopping deactivated.")

    def on_epoch_end(self, epoch, logs=None):
        if self._params.monitor not in logs:
            # Automatically determine the actual label of monitor
            if self._params.monitor.startswith("lav_"):
                logger.debug(
                    f"{self._monitor} was not found in logs. Maybe this epoch lav was not called. Skipping export best"
                )
                return

            allowed_suffixes = ["_loss", "_metric"]
            initial = self._params.monitor
            for suf in allowed_suffixes:
                if self._params.monitor + suf in logs:
                    self._params.monitor += suf
                    break

            if initial != self._params.monitor:
                logger.warning(f"{initial} was not found in logs, using {self._params.monitor} instead.")
            else:
                logger.error(f"Could not find '{self._params.monitor}' in logs: {logs}")
                return

        # Determine if a better model was found (depending on the mode)
        new_value = logs.get(self._params.monitor)
        better_found = False
        if self._params.current is None:
            better_found = True
        elif self._params.mode == "min":
            if new_value < self._params.current:
                better_found = True
        else:
            if new_value > self._params.current:
                better_found = True

        if better_found:
            # Better model: reset the number of epochs without best (n=1), store the best value, and export it.
            logger.info(
                f"Better value of {self._params.monitor} found. Old = {self._params.current}, Best = {new_value}"
            )
            self._params.current = new_value
            self._params.n = 1
            if self._export_best:
                self._scenario.export(self._export_dir, self._trainer_params)
            if self._early_stopping_enabled:
                logger.debug(f"Early stopping reset. Iteration to go = {self._params.n_to_go}")
        else:
            # No better model found
            if self._early_stopping_enabled:
                # Check if the number of epochs withoug best should increase.
                # This is only the case if the frequency is matched, i.e. this epoch counts
                # And the value is in the threshold limits
                if (epoch + 1) % self._params.frequency == 0:
                    if self._params.mode == "max" and self._params.current < self._params.lower_threshold:
                        logger.info(
                            f"Not counting {self._params.current} for early stopping since lower threshold "
                            f"{self._params.lower_threshold} was not reached"
                        )
                    elif self._params.mode == "min" and self._params.current > self._params.upper_threshold:
                        logger.debug(
                            f"Not counting {self._params.current} for early stopping since upper threshold "
                            f"{self._params.upper_threshold} was not reached"
                        )
                    else:
                        logger.info(
                            f"Early stopping progressed. (remaining iteration without improvement: "
                            f"{self._params.n_to_go - self._params.n})"
                        )
                        self._params.n += 1
                else:
                    logger.debug(f"Epoch {epoch + 1} ignored in early stopping as frequency={self._params.frequency}.")

            logger.info(
                f"No better value of {self._params.monitor} = {new_value} found. Keeping best = {self._params.current}"
            )

        # Check if training shall be stopped based on the limit thresholds
        if self._params.mode == "min":
            if self._params.current <= self._params.lower_threshold:
                self.model.stop_training = True
                logger.info(
                    f"Early stopping. Reached limit value of {self._params.monitor}: "
                    f"{self._params.current} <= {self._params.lower_threshold}"
                )
        else:
            if self._params.current >= self._params.upper_threshold:
                self.model.stop_training = True
                logger.info(
                    f"Early stopping. Reached limit value of {self._params.monitor}: "
                    f"{self._params.current} >= {self._params.upper_threshold}"
                )

        # Check if training shall be stopped based on early stopping
        if self._early_stopping_enabled and self._params.n >= self._params.n_to_go:
            self.model.stop_training = True
            logger.info(
                f"Early stopping. Reached number of maximum iterations without improvement "
                f"({self._params.n} = {self._params.n_to_go}"
            )
