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
"""Implementation of a LayerBase, a dynamic, replaceable keras.Layer

Extend both LayerBaseParams, and the LayerBase, whereby the LayerBaseParams create the LayerBase.
Use the LayerBaseParams in the ModelParams, and instantiate the actual LayerBase by calling LayerBaseParams.create().
This will create the LayerBase based on the actual replaceable params that are defined.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Type, TypeVar, Generic

from paiargparse import pai_dataclass
from tensorflow import keras
from tfaip.model.tensorboardwriter import TensorboardWriter


class TFAIPLayerBase(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tensorboard_outputs = []

    def add_tensorboard(self, tb: TensorboardWriter, value):
        if tb not in self.tensorboard_outputs:
            self.tensorboard_outputs.append(tb)

        self.add_metric(tb(None, value))


@pai_dataclass
@dataclass
class LayerBaseParams(ABC):
    @staticmethod
    @abstractmethod
    def cls() -> Type["LayerBase"]:
        raise NotImplementedError

    def create(self, **kwargs):
        return self.cls()(params=self, **kwargs)


TLP = TypeVar("TLP", bound=LayerBaseParams)


class LayerBase(Generic[TLP], TFAIPLayerBase):
    """
    This Layer can be inherited to buildup the graph (you can however chose any method you want to).
    """

    @classmethod
    def params_cls(cls):
        arg = cls.__orig_bases__[0].__args__[0]
        if isinstance(arg, TypeVar):
            return arg.__bound__  # default
        return arg

    def __init__(self, params: TLP, **kwargs):
        super().__init__(**kwargs)
        self._params = params

    @property
    def params(self) -> TLP:
        return self._params

    def get_config(self):
        cfg = super().get_config()
        cfg["params"] = self._params.to_dict()
        return cfg

    @classmethod
    def from_config(cls, config):
        config["params"] = cls.params_cls().from_dict(config["params"])
        return super(LayerBase, cls).from_config(config)
