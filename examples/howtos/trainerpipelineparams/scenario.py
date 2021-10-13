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
"""Example to show how to implement a custom TrainerPipelineParams.

This example shows how to share parameters between a train and val data generator.
"""
from dataclasses import dataclass
from typing import Dict, Type, Iterable, Union, List, Optional

import tensorflow as tf
from paiargparse import pai_dataclass

from tfaip import (
    ModelBaseParams,
    DataBaseParams,
    ScenarioBaseParams,
    TrainerPipelineParamsBase,
    DataGeneratorParams,
    Sample,
)
from tfaip.data.data import DataBase
from tfaip.data.pipeline.datagenerator import DataGenerator
from tfaip.model.modelbase import ModelBase
from tfaip.scenario.scenariobase import ScenarioBase
from tfaip.util.tftyping import AnyTensor


# =====================================================
# Trainer pipeline params definition
@pai_dataclass
@dataclass
class MyDataGeneratorParams(DataGeneratorParams):
    gen_from: int
    gen_to: int

    @staticmethod
    def cls() -> Type["DataGenerator"]:
        return MyDataGenerator


class MyDataGenerator(DataGenerator[MyDataGeneratorParams]):
    """A Dummy DataGenerator that yields some numbers"""

    def __len__(self):
        return self.params.gen_to - self.params.gen_from

    def generate(self) -> Iterable[Union[Sample, List[Sample]]]:
        return range(self.params.gen_from, self.params.gen_to)


@pai_dataclass
@dataclass
class MyPipelineWithIdenticalTrainAndValData(TrainerPipelineParamsBase[MyDataGeneratorParams, MyDataGeneratorParams]):
    count: int = 100
    amount_val: float = 0.2

    def train_gen(self) -> MyDataGeneratorParams:
        return MyDataGeneratorParams(int(self.count * self.amount_val), self.count)

    def val_gen(self) -> Optional[MyDataGeneratorParams]:
        return MyDataGeneratorParams(0, int(self.count * self.amount_val))


# =====================================================
# Data definition


@pai_dataclass
@dataclass
class MyModelParams(ModelBaseParams):
    def graph_cls(self):
        raise NotImplementedError  # Implement the actual graph

    @staticmethod
    def cls():
        return MyModel


class MyModel(ModelBase[MyModelParams]):
    def _loss(self, inputs, targets, outputs) -> Dict[str, AnyTensor]:
        raise NotImplementedError


# =====================================================
# Data definition


@pai_dataclass
@dataclass
class MyDataParams(DataBaseParams):
    @staticmethod
    def cls() -> Type["DataBase"]:
        return MyDataBase


class MyDataBase(DataBase[MyDataParams]):
    def _input_layer_specs(self) -> Dict[str, tf.TensorSpec]:
        raise NotImplementedError

    def _target_layer_specs(self) -> Dict[str, tf.TensorSpec]:
        raise NotImplementedError


@pai_dataclass
@dataclass
class MyScenarioParams(ScenarioBaseParams[MyDataParams, MyModelParams]):
    ...


class MyScenario(ScenarioBase[MyScenarioParams, MyPipelineWithIdenticalTrainAndValData]):
    ...
