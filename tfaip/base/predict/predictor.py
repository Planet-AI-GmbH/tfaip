# Copyright 2020 The tfaip authors. All Rights Reserved.
#
# This file is part of tfaip.
#
# tfaip is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# tfaip is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# tfaip. If not, see http://www.gnu.org/licenses/.
# ==============================================================================
from typing import Union, TYPE_CHECKING, Iterable

from tensorflow import keras

from tfaip.base.data.pipeline.definitions import Sample
from tfaip.base.predict.predictorbase import PredictorBase, PredictorParams

if TYPE_CHECKING:
    from tfaip.base.data.data import DataBase


class Predictor(PredictorBase):
    def __init__(self, params: PredictorParams, data: 'DataBase'):
        super(Predictor, self).__init__(params, data)

    def set_model(self, model: Union[str, keras.Model]):
        self._keras_model = self._load_model(model)

    @property
    def model(self):
        return self._keras_model

    def _unwrap_batch(self, inputs, targets, outputs) -> Iterable[Sample]:
        batch_size = next(iter(outputs.values())).shape[0]
        for i in range(batch_size):
            un_batched_outputs = {k: v[i] for k, v in outputs.items()}
            un_batched_inputs = {k: v[i] for k, v in inputs.items()}
            un_batched_targets = {k: v[i] for k, v in targets.items()}
            sample = Sample(inputs=un_batched_inputs, outputs=un_batched_outputs, targets=un_batched_targets)

            yield sample

    def _print_prediction(self, sample: Sample, print_fn):
        print_fn(f"\n     PREDICTION:\n" + "\n".join([f'        {k}: mean = {v.mean()}, max = {v.max()}, min = {v.min()}' for k, v in sample.outputs.items()]))

