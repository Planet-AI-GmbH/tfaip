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
from abc import abstractmethod

import tensorflow.keras as keras

from tfaip.base.model import ModelBaseParams


class GraphBase(keras.layers.Layer):
    """
    This Layer can be inherited to buildup the graph (you can however chose any method you want to).
    """

    @classmethod
    @abstractmethod
    def params_cls(cls):
        raise NotImplemented

    def __init__(self, params: 'ModelBaseParams', **kwargs):
        super(GraphBase, self).__init__(**kwargs)
        self._params = params

    def get_config(self):
        cfg = super(GraphBase, self).get_config()
        cfg['params'] = self._params.to_dict()
        return cfg

    @classmethod
    def from_config(cls, config):
        config['params'] = cls.params_cls().from_dict(config['params'])
        return super(GraphBase, cls).from_config(config)
