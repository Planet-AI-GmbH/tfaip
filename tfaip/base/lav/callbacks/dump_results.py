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
