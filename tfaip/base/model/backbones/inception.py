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
from dataclasses import dataclass, field
from typing import Union, Optional

import tensorflow.keras as keras
from dataclasses_json import dataclass_json

from tfaip.base.imports import ModelBaseParams
from tfaip.base.model.backbones import BackboneModel


@dataclass_json
@dataclass
class InceptionV3Params(ModelBaseParams):
    weights: str = field(default=None)
    # Global pooling on the last conv feature map
    pooling: Optional[str] = field(default=None)
    # The following parameter are just of interest in case of classification
    include_top: bool = field(default=False)
    classes: int = field(default=1000)
    classifier_activation: Union[str, callable] = field(default="softmax")


class InceptionV3(BackboneModel):
    def __init__(self, params: InceptionV3Params):
        """
        Here is a mapping from the old_names to the new names:
        Old name          | tf_aip name     | tf2_aip name    | subsampling
        ================================================================
        conv0             | Conv2d_1a_3x3   | activation      | 2
        conv1             | Conv2d_2a_3x3   | activation_1    | 2
        conv2             | Conv2d_2b_3x3   | activation_2    | 2
        pool1             | MaxPool_3a_3x3  | max_pooling2d   | 4
        conv3             | Conv2d_3b_1x1   | activation_3    | 4
        conv4             | Conv2d_4a_3x3   | activation_4    | 4
        pool2             | MaxPool_5a_3x3  | max_pooling2d_1 | 8
        mixed_35x35x256a  | Mixed_5b        | mixed0          | 8
        mixed_35x35x288a  | Mixed_5c        | mixed1          | 8
        mixed_35x35x288b  | Mixed_5d        | mixed2          | 16
        mixed_17x17x768a  | Mixed_6a        | mixed3          | 16
        mixed_17x17x768b  | Mixed_6b        | mixed4          | 16
        mixed_17x17x768c  | Mixed_6c        | mixed5          | 16
        mixed_17x17x768d  | Mixed_6d        | mixed6          | 16
        mixed_17x17x768e  | Mixed_6e        | mixed7          | 16
        mixed_8x8x1280a   | Mixed_7a        | mixed8          | 32
        mixed_8x8x2048a   | Mixed_7b        | mixed9          | 32
        mixed_8x8x2048b   | Mixed_7c        | mixed10         | 32
        ---               | ---             | predictions     | ---

        The layer "predictions" is only included if include_top is True.
        """

        super().__init__(params)

    def build(self, input_shape):
        # TODO include gray to rgb, into model in case of pretrained weights
        # Here take care that last channel equals 3
        self._model = keras.applications.InceptionV3(**self._params.__dict__, input_shape=input_shape['image'][1:])

    def call(self, inputs, **kwargs):
        # Here perform the image transformation
        return self._model(inputs, **kwargs)

    @classmethod
    def params_cls(cls):
        return InceptionV3Params


if __name__ == '__main__':
    instance = InceptionV3(InceptionV3Params())
    instance.build({'image': [None, 299, 299, 3]})
    instance._model.summary()
