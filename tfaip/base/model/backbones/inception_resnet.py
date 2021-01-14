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
class InceptionResNetParams(ModelBaseParams):
    weights: str = field(default=None)
    # Global pooling on the last conv feature map
    pooling: Optional[str] = field(default=None)
    # The following parameter are just of interest in case of classification
    include_top: bool = field(default=False)
    classes: int = field(default=1000)
    classifier_activation: Union[str, callable] = field(default="softmax")


class InceptionResNet(BackboneModel):
    def __init__(self, params: InceptionResNetParams):
        """
        Here is a mapping from the old_names to the new names:
        name             |  ss | comment
        ==================================================================
        activation_4     |   4 | stem before maxpooling
        max_pooling2d_1  |   8 | stem after maxpooling
        mixed_5b         |   8 | Inception-A block
        block35_<IDX>_ac |   8 | Inception-ResNet-A block IDX in [1,..,10]
        mixed_6a         |  16 | Reduction-A block
        block17_<IDX>_ac |  16 | Inception-ResNet-B block IDX in [1,..,20]
        mixed_7a         |  32 | Reduction-B block
        block8_<IDX>_ac  |  32 | Inception-ResNet-C block IDX in [1,..,9]
        block8_10        |  32 | Inception-ResNet-C block of Index 10
        conv_7b_ac       |  32 | final convolutional with activation
        predictions      | --- | with average pooling

        The layer "predictions" is only included if include_top is True.
        """

        super().__init__(params)

    def build(self, input_shape):
        # TODO include gray to rgb, into model in case of pretrained weights
        # Here take care that last channel equals 3
        self._model = keras.applications.InceptionResNetV2(**self._params.__dict__,
                                                           input_shape=input_shape['image'][1:])

    def call(self, inputs, **kwargs):
        # Here perform the image transformation
        return self._model(inputs, **kwargs)

    @classmethod
    def params_cls(cls):
        return InceptionResNetParams


if __name__ == '__main__':
    params = InceptionResNetParams()
    # params.include_top=True
    instance = InceptionResNet(params)
    instance.build({'image': [None, 299, 299, 3]})
    instance._model.summary()
