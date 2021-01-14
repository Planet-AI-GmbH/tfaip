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
from abc import abstractmethod, ABC
from typing import Optional, List

import tensorflow.keras as keras

from tfaip.base.imports import GraphBase, ModelBaseParams


class BackboneModel(GraphBase, ABC):

    def __init__(self, params: 'ModelBaseParams', **kwargs):
        super().__init__(params, **kwargs)
        self._model: Optional[keras.Model] = None

    @classmethod
    @abstractmethod
    def params_cls(cls):
        raise NotImplementedError

    @property
    def inputs(self):
        return self._model.inputs

    @property
    def layers(self):
        return self._model.layers

    def get_layer(self, name=None, index=None):
        return self._model.get_layer(name, index)

