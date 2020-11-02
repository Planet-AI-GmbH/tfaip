import tensorflow.keras.callbacks as cb


# See https://github.com/tensorflow/tensorflow/issues/42872
class TensorflowFix(cb.Callback):
    def __init__(self):
        super(TensorflowFix, self).__init__()
        self._supports_tf_logs = True       # Any Callback before LAV callback must act on raw tf logs only
        self._backup_loss = None

    def on_train_begin(self, logs=None):
        self._backup_loss = {**self.model.loss}

    def on_train_batch_end(self, batch, logs=None):
        self.model.loss = self._backup_loss
