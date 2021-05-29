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
"""Definition of the LAVCallback
"""
import logging
import os
import time
from typing import TYPE_CHECKING

import numpy as np
from tensorflow.keras.callbacks import Callback

if TYPE_CHECKING:
    from tfaip.trainer.callbacks.extract_logs import ExtractLogsCallback
    from tfaip.imports import TrainerParams, ScenarioBase

logger = logging.getLogger(__name__)


class LAVCallback(Callback):
    """This callback runs LAV at the end of a epoch.

    All output metrics of LAV are added to the logs (prefix lav_) and can thus be accessed in other callbacks.
    Therefore, LAV results are also added to the tensorboard (with a custom LAV handler)
    """

    def __init__(
        self, trainer_params: "TrainerParams", scenario: "ScenarioBase", extract_logs_cb: "ExtractLogsCallback"
    ):
        super().__init__()
        self._supports_tf_logs = True  # True so that we can manipulate the logs
        self.scenario = scenario
        self.trainer_params = trainer_params
        lav_params = scenario.lav_cls().params_cls()()
        lav_params.device = trainer_params.device
        lav_params.model_path = os.getcwd()  # Here resources are still relative to current working dir
        lav_params.silent = True
        lav_params.pipeline = trainer_params.gen.setup.val
        self.lav = scenario.create_lav(lav_params=lav_params, scenario_params=scenario.params)
        self.lav_this_epoch = False
        self.extract_logs_cb = extract_logs_cb

    def on_epoch_begin(self, epoch, logs=None):
        # Determine if LAV should be run in this epoch
        if epoch < self.trainer_params.lav_min_epoch:
            self.lav_this_epoch = False
        else:
            self.lav_this_epoch = (
                epoch % self.trainer_params.lav_every_n
            ) == 0 or epoch == self.trainer_params.epochs - 1

    def on_epoch_end(self, epoch, logs=None):
        # Run LAV
        if not self.lav_this_epoch:
            logger.debug(f"No LAV in epoch {epoch}")
            return

        logger.info("Running LAV")
        start = time.time()
        logs = logs if logs else {}
        for i, r in enumerate(
            self.lav.run(
                self.trainer_params.gen.lav_gen(),
                self.scenario.keras_predict_model,
                run_eagerly=self.trainer_params.force_eager,
                return_tensorboard_outputs=True,
            )
        ):
            logs_str = " - ".join(f"{k}: {np.mean(r[k]):.4f}" for k in sorted(r.keys()) if not isinstance(r[k], bytes))
            logs_str = f"LAV l{i} Metrics (dt={(time.time() - start) / 60:.2f}min) - {logs_str}"
            logger.info(logs_str)
            for k, v in r.items():
                if "multi_metric" in k:
                    continue
                if self.extract_logs_cb.tensorboard_data_handler.is_tensorboard_only(k, v):
                    self.extract_logs_cb.extracted_logs[f"lav_l{i}_{k}"] = v
                else:
                    logs[f"lav_l{i}_{k}"] = v
