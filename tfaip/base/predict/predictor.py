from typing import Union, TYPE_CHECKING, Iterable, Type

from tensorflow import keras

from tfaip.base.data.pipeline.definitions import InputOutputSample
from tfaip.base.predict.predictorbase import PredictorBase, PredictorParams

if TYPE_CHECKING:
    from tfaip.base.data.data import DataBase


class Predictor(PredictorBase):
    def __init__(self, params: PredictorParams, data: 'DataBase'):
        super(Predictor, self).__init__(params, data)

    def set_model(self, model: Union[str, keras.Model]):
        self._keras_model = self._load_model(model)

    def _unwrap_batch(self, inputs, outputs) -> Iterable:
        batch_size = next(iter(outputs.values())).shape[0]
        for i in range(batch_size):
            un_batched_outputs = {k: v[i] for k, v in outputs.items()}
            un_batched_inputs = {k: v[i] for k, v in inputs.items()}
            sample = InputOutputSample(un_batched_inputs, un_batched_outputs)

            yield sample

    def _print_prediction(self, sample: InputOutputSample, print_fn):
        print_fn(f"\n     PREDICTION:\n" + "\n".join([f'        {k}: mean = {v.mean()}, max = {v.max()}, min = {v.min()}' for k, v in sample.outputs.items()]))

