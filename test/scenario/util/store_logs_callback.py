from tensorflow.keras.callbacks import Callback


class StoreLogsCallback(Callback):
    def __init__(self):
        super(StoreLogsCallback, self).__init__()
        self.logs = {}

    def on_epoch_end(self, epoch, logs=None):
        self.logs = logs


