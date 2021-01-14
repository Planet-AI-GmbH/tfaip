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
"""EfficientNet models for Keras.

Reference paper:
  - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks]
    (https://arxiv.org/abs/1905.11946) (ICML 2019)
"""

__author__ = "Gundram Leifert"
__copyright__ = "Copyright 2020, Planet AI GmbH"
__credits__ = ["Gundram Leifert"]
__email__ = "gundram.leifert@planet-ai.de"

import inspect
from dataclasses import dataclass, field
from typing import Union, Optional

import tensorflow.keras.applications.efficientnet as file_efficientnet
from dataclasses_json import dataclass_json

from tfaip.base.imports import ModelBaseParams
from tfaip.base.model.backbones import BackboneModel
from tfaip.util.argumentparser import dc_meta


@dataclass_json
@dataclass
class EfficientNetParams(ModelBaseParams):
    weights: str = field(default="imagenet")
    # Global pooling on the last conv feature map
    pooling: Optional[str] = field(default=None)
    # The following parameter are just of interest in case of classification
    include_top: bool = field(default=False)
    classes: int = field(default=1000)
    classifier_activation: Union[str, callable] = field(default="softmax")
    mode: int = field(default=3, metadata=dc_meta(help='one of [0,1,2,3,4,5]'))


class EfficientNet(BackboneModel):

    def __init__(self, params: EfficientNetParams):
        """
        Here is a mapping from the old_names to the new names:
        name                        | subsampling
        ==========================================
        stem_conv                   | 2
        block2a_expand_activation   | 2
        block3a_expand_activation   | 4
        block4a_expand_activation   | 8
        block5a_expand_activation   | 16
        block6a_expand_activation   | 16
        block7a_expand_activation   | 32
        top_activation              | 32
        predictions                 | ---

        The layer "predictions" is only included if include_top is True.
        """
        super().__init__(params)
        if self._params.mode not in range(8):
            raise Exception(f"parameter 'mode' not in range [0..7], it is {self._params.mode}")
        self._model_fn = getattr(file_efficientnet, f'EfficientNetB{self._params.mode}')
        p = inspect.getfullargspec(self._model_fn)
        self._reduced_dict = {k: v for k, v in self._params.__dict__.items() if k in p.args}

    def build(self, input_shape):
        # TODO include gray to rgb, into model in case of pretrained weights
        # Here take care that last channel equals 3
        shape = input_shape['image'][1:] if input_shape is not None and 'image' in input_shape and input_shape[
            'image'] is not None else None
        self._model = self._model_fn(**self._reduced_dict, input_shape=shape)

    def call(self, inputs, **kwargs):
        # Here perform the image transformation
        return self._model(inputs, **kwargs)

    @classmethod
    def params_cls(cls):
        return EfficientNetParams


if __name__ == '__main__':
    params = EfficientNetParams()
    params.mode = 7
    # params.include_top = True
    instance = EfficientNet(params)
    instance.build({'image': [None, 224, 224, 3]})
    instance._model.summary()
