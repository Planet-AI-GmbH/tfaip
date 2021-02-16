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
import time

from tensorflow.keras.callbacks import Callback
from typing import TYPE_CHECKING
import logging
import os

if TYPE_CHECKING:
    from tfaip.base.trainer.trainerparams import TrainerParams
    from tfaip.base.scenario.scenariobase import ScenarioBase


logger = logging.getLogger(__name__)


class LAVCallback(Callback):
    def __init__(self, trainer_params: 'TrainerParams', scenario: 'ScenarioBase'):
        super(LAVCallback, self).__init__()
        self._supports_tf_logs = True       # True so that we can manipulate the logs
        self.scenario = scenario
        self.trainer_params = trainer_params
        lav_params = scenario.lav_cls().get_params_cls()()
        lav_params.device_params = trainer_params.device_params
        lav_params.model_path_ = os.getcwd()  # Here resources are still relative to current working dir
        lav_params.silent = True
        self.lav = scenario.create_lav(lav_params=lav_params, scenario_params=scenario.params)
        self.lav_this_epoch = False

    def on_epoch_begin(self, epoch, logs=None):
        self.lav_this_epoch = (epoch % self.trainer_params.lav_every_n) == 0 or epoch == self.trainer_params.epochs - 1

    def on_epoch_end(self, epoch, logs=None):
        if not self.lav_this_epoch:
            logger.debug(f"No LAV in epoch {epoch}")
            return

        logger.info("Running LAV")
        start = time.time()
        logs = logs if logs else {}
        for i, r in enumerate(self.lav.run(self.scenario.keras_predict_model)):
            logs_str = ' - '.join(f"{k}: {r[k]:.4f}" for k in sorted(r.keys()))
            logger.info(f"LAV l{i} Metrics (dt={(time.time() - start)/60:.2f}min) - {logs_str}")
            for k, v in r.items():
                logs[f"lav_l{i}_{k}_metric"] = v
