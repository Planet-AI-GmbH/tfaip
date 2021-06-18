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
"""Definition of the TFAIPProgbarLogger which extends the default keras ProgbarLogger"""
import time

from tensorflow.python.keras.callbacks import ProgbarLogger
from tfaip.util.tftyping import sync_to_numpy_or_python_type


class TFAIPProgbarLogger(ProgbarLogger):
    """
    Callback to render the progress bar during trainer.
    This implementation of the default ProgbarLogger ads an additional mode (self.verbose == 2),
    Where instead of a progress bar the output is logged each delta_time seconds (default 5).
    """

    def __init__(self, delta_time=5, **kwargs):
        super().__init__(**kwargs)
        self._time_remaining = 0
        self._delta_time = delta_time  # Output every 5 secs, by default
        self._last_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self._last_time = time.time()
        self._time_remaining = 0
        super().on_epoch_begin(epoch, logs)

    def _batch_update_progbar(self, batch, logs=None):
        super()._batch_update_progbar(batch, logs)
        if self.verbose == 2:
            if self._time_remaining <= 0:
                self._time_remaining += self._delta_time
                self._last_time = time.time()
                # Only block async when verbose = 1.
                logs = sync_to_numpy_or_python_type(logs)
                self.progbar.update(self.seen, list(logs.items()), finalize=True)
            else:
                new_time = time.time()
                self._time_remaining -= new_time - self._last_time
                self._last_time = new_time
