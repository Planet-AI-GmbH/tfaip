from tfaip import Sample
from tfaip.predict.predictor import Predictor

from examples.imageclassification.params import Keys


class ICPredictor(Predictor):
    def _print_prediction(self, sample: Sample, print_fn):
        class_name = self.data.params.classes[sample.outputs[Keys.OutputClass]]
        conf = sample.outputs[Keys.OutputSoftmax][sample.outputs[Keys.OutputClass]]
        print_fn(f"This image most likely belongs to {class_name} with a {conf:.2f} percent confidence.")
