import time

from tensorflow.python.keras.callbacks import ProgbarLogger
from tensorflow.python.keras.utils.tf_utils import to_numpy_or_python_type


class TFAIPProgbarLogger(ProgbarLogger):
    def __init__(self, delta_time=5, **kwargs):
        super(TFAIPProgbarLogger, self).__init__(**kwargs)
        self._time_remaining = 0
        self._delta_time = delta_time  # Output every 5 secs, by default
        self._last_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self._last_time = time.time()
        self._time_remaining = 0
        super(TFAIPProgbarLogger, self).on_epoch_begin(epoch, logs)

    def _batch_update_progbar(self, batch, logs=None):
        super(TFAIPProgbarLogger, self)._batch_update_progbar(batch, logs)
        if self.verbose == 2:
            if self._time_remaining <= 0:
                self._time_remaining += self._delta_time
                self._last_time = time.time()
                # Only block async when verbose = 1.
                logs = to_numpy_or_python_type(logs)
                self.progbar.update(self.seen, list(logs.items()), finalize=True)
            else:
                new_time = time.time()
                self._time_remaining -= new_time - self._last_time
                self._last_time = new_time
