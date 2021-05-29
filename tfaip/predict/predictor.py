# Copyright 2021 The tfaip authors. All Rights Reserved.
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
"""Definition of the Predictor"""
import json
from typing import Union, Iterable

from tensorflow import keras
import tensorflow as tf

from tfaip import Sample
from tfaip.predict.predictorbase import PredictorBase
from tfaip.util.json_helper import TFAIPJsonDecoder


class Predictor(PredictorBase):
    """
    Predictor that applies a single model on data. This is the most common case.

    For usage either call Scenario.create_predictor or instantiate the Predictor directly and then call set_model.
    """

    def set_model(self, model: Union[str, keras.Model]):
        self._keras_model = self._load_model(model)

    @property
    def model(self):
        return self._keras_model

    def _unwrap_batch(self, inputs, targets, outputs, meta) -> Iterable[Sample]:
        try:
            batch_size = next(iter(outputs.values())).shape[0]
        except StopIteration as e:
            raise ValueError(f"Outputs are empty: {outputs}") from e

        for i in range(batch_size):
            un_batched_outputs, un_batched_inputs, un_batched_targets, un_batched_meta = tf.nest.map_structure(
                lambda x: x[i], (outputs, inputs, targets, meta)
            )
            parsed_meta = json.loads(un_batched_meta["meta"][0].decode("utf-8"), cls=TFAIPJsonDecoder)
            sample = Sample(
                inputs=un_batched_inputs, outputs=un_batched_outputs, targets=un_batched_targets, meta=parsed_meta
            )

            yield sample

    def _print_prediction(self, sample: Sample, print_fn):
        print_fn(
            "\n     PREDICTION:\n"
            + "\n".join(
                [f"        {k}: mean = {v.mean()}, max = {v.max()}, min = {v.min()}" for k, v in sample.outputs.items()]
            )
        )
