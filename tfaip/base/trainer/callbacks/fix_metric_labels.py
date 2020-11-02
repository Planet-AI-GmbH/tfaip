import tensorflow.keras.callbacks as cb


class FixMetricLabelsCallback(cb.Callback):
    def __init__(self):
        super(FixMetricLabelsCallback, self).__init__()
        self._supports_tf_logs = True
        self.original_metrics = {}

    def on_train_begin(self, logs=None):
        # store original metric names
        self.original_metrics = self.model.compiled_metrics._weighted_metrics

    def on_epoch_end(self, epoch, logs=None):
        self.fix(logs)

    def on_train_batch_end(self, batch, logs=None):
        self.fix(logs)

    def on_predict_batch_end(self, batch, logs=None):
        self.fix(logs)

    def on_test_batch_end(self, batch, logs=None):
        self.fix(logs)

    def fix(self, logs):
        if logs is None:
            return

        for n, m in self.original_metrics.items():
            if not m or not hasattr(m, 'name') or m.name == n:
                continue
            if m.name in logs and n not in logs:
                logs[n] = logs[m.name]
                del logs[m.name]

