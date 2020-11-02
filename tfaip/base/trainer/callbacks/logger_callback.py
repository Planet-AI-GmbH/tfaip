from tensorflow.keras.callbacks import Callback
import logging


logger = logging.getLogger(__name__)


class LoggerCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        logs_str = ' - '.join(f"{k}: {logs[k]:.4f}" for k in sorted(logs.keys()))
        logger.info(f"Results of epoch {epoch:4d}: {logs_str}")
