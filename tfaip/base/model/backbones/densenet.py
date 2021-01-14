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
__author__ = "Gundram Leifert"
__copyright__ = "Copyright 2020, Planet AI GmbH"
__credits__ = ["Gundram Leifert"]
__email__ = "gundram.leifert@planet-ai.de"

import inspect
from dataclasses import dataclass, field
from typing import Optional, Union, List

from dataclasses_json import dataclass_json
from tensorflow import keras

from tfaip.base.imports import ModelBaseParams
from tfaip.base.model.backbones import BackboneModel
from tfaip.util.argumentparser import dc_meta


@dataclass_json
@dataclass
class DenseNetParams(ModelBaseParams):
    weights: str = field(default="imagenet")
    # Global pooling on the last conv feature map
    pooling: Optional[str] = field(default=None)
    # The following parameter are just of interest in case of classification
    include_top: bool = field(default=False)
    classes: int = field(
        default=1000,
        metadata=dc_meta(help="only has influece if 'include_top' == True"))
    classifier_activation: Union[str, callable] = field(
        default="softmax",
        metadata=dc_meta(help="only has influece if 'include_top' == True"))
    mode: Union[int, List[int]] = field(default_factory=lambda: 121, metadata=dc_meta(help='one of [121,169,201]'))


class DenseNet(BackboneModel):
    def __init__(self, params: DenseNetParams):
        """
        Here is a mapping from the old_names to the new names:
        name        | subsampling | comment
        ================================================================
        conv1/relu  | 2           | only stem
        pool2_relu  | 4           | with 1 dense block (grow = 24)
        pool3_relu  | 8           | with 2 dense blocks (grow = 24)
        pool4_relu  | 16          | with 3 dense blocks (grow = 24)
        relu        | 32          | with 4 dense blocks (grow = 24)
        predictions | ---         | avg_pooling + dense(#classes)

        The layer "predictions" is only included if include_top is True.
        """

        super().__init__(params)
        if isinstance(params.mode, int):
            assert params.mode in [121, 169, 201]
        else:
            assert isinstance(params.mode, list)
            assert len(params.mode) in [1, 4]

    def build(self, input_shape):
        # TODO include gray to rgb, into model in case of pretrained weights
        # Here take care that last channel equals 3
        model = {
            121: keras.applications.densenet.DenseNet121,
            169: keras.applications.densenet.DenseNet169,
            201: keras.applications.densenet.DenseNet201,
        }
        params = self._params.__dict__
        if isinstance(self._params.mode, int):
            model_fn = model[self._params.mode]
        else:
            if len(self._params.mode) == 1:
                model_fn = model[self._params.mode]
            else:
                from tensorflow.python.keras.applications.densenet import DenseNet
                model_fn = DenseNet
                params['blocks'] = self._params.mode
        p = inspect.getfullargspec(model_fn)
        reduced_dict = {k: v for k, v in self._params.__dict__.items() if k in p.args}
        self._model = model_fn(**reduced_dict, input_shape=input_shape['image'][1:])

    def call(self, inputs, **kwargs):
        # Here perform the image transformation
        return self._model(inputs, **kwargs)

    @classmethod
    def params_cls(cls):
        return DenseNetParams


if __name__ == '__main__':
    params: DenseNetParams = DenseNetParams()
    own_size = False
    if own_size:
        params.mode = [4, 4, 4, 4]
        params.weights = None
    else:
        params.mode = 121
    # params.include_top=True
    instance = DenseNet(params)
    instance.build({'image': [None, 224, 224, 3]})
    instance._model.summary()
