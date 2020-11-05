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
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dataclasses_json import dataclass_json
from tensorflow.keras.callbacks import Callback

if TYPE_CHECKING:
    from tfaip.base.scenario.scenariobase import ScenarioBase
    from tfaip.base.trainer.trainer import TrainerParams

logger = logging.getLogger(__name__)


@dataclass_json
@dataclass
class ExportBestState:
    mode: str = None
    current: float = None  # If loaded from a checkpoint, this value is already set
    monitor: str = None


class ExportBestCallback(Callback):
    def __init__(self, export_dir, scenario: 'ScenarioBase', trainer_params: 'TrainerParams'):
        # mode = min or max
        super(ExportBestCallback, self).__init__()
        self._export_dir = export_dir
        self._scenario = scenario
        self._trainer_params = trainer_params
        self._state = trainer_params.export_best_state_

        if self._state.monitor is None or self._state.mode is None:
            mode, monitor = scenario.best_logging_settings()
            mode = mode.lower()
            if not monitor.startswith("val_"):
                logger.info(f"Automatically selecting 'val_{monitor}' instead of '{monitor}' for monitoring.")
                monitor = "val_" + monitor

            assert mode in ["min", "max"]
            self._state.monitor, self._state.mode = monitor, mode
        else:
            # Already defined from loaded checkpoint (e.g. resume training)
            pass

    def on_epoch_end(self, epoch, logs=None):
        if self._state.monitor not in logs:
            if self._state.monitor.startswith("lav_"):
                logger.debug(
                    f"{self._monitor} was not found in logs. Maybe this epoch lav was not called. Skipping export best")
                return

            allowed_suffixes = ['_loss', '_metric']
            initial = self._state.monitor
            for suf in allowed_suffixes:
                if self._state.monitor + suf in logs:
                    self._state.monitor += suf
                    break

            if initial != self._state.monitor:
                logger.warning("{} was not found in logs, using {} instead.".format(initial, self._state.monitor))
            else:
                logger.error("Could not find '{}' in logs: {}".format(self._state.monitor, logs))
                return

        new_value = logs.get(self._state.monitor)
        better_found = False
        if self._state.current is None:
            better_found = True
        elif self._state.mode == 'min':
            if new_value < self._state.current:
                better_found = True
        else:
            if new_value > self._state.current:
                better_found = True

        if better_found:
            logger.info("Better value of {} found. Old = {}, Best = {}".format(self._state.monitor, self._state.current,
                                                                               new_value))
            self._state.current = new_value
            self._scenario.export(self._export_dir, self._trainer_params)
        else:
            logger.info("No better value of {} found. Worse = {}, Best = {}".format(self._state.monitor, new_value,
                                                                                    self._state.current))
