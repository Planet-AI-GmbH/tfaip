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
from abc import ABC, abstractmethod

from tfaip.base.model import GraphBase
import tensorflow.keras.backend as K


class TutorialGraph(GraphBase, ABC):
    @classmethod
    def params_cls(cls):
        from tfaip.scenario.tutorial.model import ModelParams
        return ModelParams

    def __init__(self, params, **kwargs):
        super(TutorialGraph, self).__init__(params, **kwargs)

    def call(self, inputs, **kwargs):
        # call function that is shared by all other graphs
        rescaled_img = K.expand_dims(K.cast(inputs['img'], dtype='float32') / 255, -1)
        logits = self._call(rescaled_img)  # call the actual graph (MLP or CNN)
        pred = K.softmax(logits, axis=-1)
        cls = K.argmax(pred, axis=-1)
        return {'pred': pred, 'logits': logits, 'class': cls}

    @abstractmethod
    def _call(self, inputs, **kwargs):
        # reimplement by actual graph
        raise NotImplementedError
