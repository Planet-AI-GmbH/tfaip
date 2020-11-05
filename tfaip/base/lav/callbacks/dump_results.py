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
from tfaip.base.lav.callbacks.lav_callback import LAVCallback
import pickle
import logging

logger = logging.getLogger(__name__)


class DumpResultsCallback(LAVCallback):
    def __init__(self, filepath: str):
        super(DumpResultsCallback, self).__init__()
        self.target_prediction_pairs = []
        self.filepath = filepath
        logger.info(f"Results dumper to {self.filepath} created.")

    def on_sample_end(self, inputs, targets, outputs):
        targets, prediction = self.model.target_prediction(targets, outputs, self.data)
        self.target_prediction_pairs.append((targets, prediction))

    def on_step_end(self, inputs, targets, outputs, metrics):
        pass

    def on_lav_end(self, result):
        logger.info(f"Dumping results {self.filepath}.")
        with open(self.filepath, 'wb') as f:
            pickle.dump({'target_prediction_pairs': self.target_prediction_pairs,
                         'result': result}, f)
