from tensorflow.keras.callbacks import Callback
import logging

from typeguard import typechecked

from tfaip.base.trainer.optimizer.weights_moving_average import WeightsMovingAverage


logger = logging.getLogger(__name__)


class EMACallback(Callback):
    @typechecked
    def __init__(self, optimizer: WeightsMovingAverage):
        # any callback after this one will have ema weights on epoch end
        # (useful for exporting best, but not checkpointing)
        super(EMACallback, self).__init__()
        self.optimizer = optimizer

    def on_test_begin(self, logs=None):
        self.optimizer.to_avg(self.model.variables)

    def on_test_end(self, logs=None):
        self.optimizer.to_model(self.model.variables)

    def on_epoch_begin(self, epoch, logs=None):
        self.optimizer.to_model(self.model.variables)

    def on_epoch_end(self, epoch, logs=None):
        self.optimizer.to_avg(self.model.variables)
