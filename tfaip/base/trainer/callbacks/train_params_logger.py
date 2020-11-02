import json
import os

import tensorflow.keras as keras
import logging


logger = logging.getLogger(__name__)


class TrainParamsLoggerCallback(keras.callbacks.Callback):
    def __init__(self, train_params, log_dir=None):
        super(TrainParamsLoggerCallback, self).__init__()
        self.log_dir = log_dir
        self.train_params = train_params

    def _save(self):
        if self.log_dir:
            path = os.path.join(self.log_dir, 'trainer_params.json')
            logger.debug("Logging current trainer state to '{}'".format(path))
            with open(path, 'w') as f:
                json.dump(self.train_params.to_dict(), f, indent=2)

    def on_train_begin(self, logs=None):
        self._save()

    def on_epoch_end(self, epoch, logs=None):
        # we ended an epoch, store that we ended it
        self.train_params.current_epoch = epoch + 1
        self._save()

