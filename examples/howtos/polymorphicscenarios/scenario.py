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
"""Example for how to setup generic scenarios with polymorphism.

Here, we create a new base Scenario for our use-case `MyScenario` and create two variants with different
models `MyModelAParams`, `MyModelBParams`. In this example the `DataBaseParams` are fixed to `MyDataParams`.

"""

from abc import ABC
from dataclasses import dataclass
from typing import TypeVar, Type, Dict

import tensorflow as tf
from paiargparse import pai_dataclass

from tfaip import ModelBaseParams, ScenarioBaseParams, DataBaseParams
from tfaip.data.data import DataBase
from tfaip.model.modelbase import ModelBase
from tfaip.scenario.listfile.listfilescenario import ListFileScenario

# =====================================================
# Model definition
from tfaip.util.tftyping import AnyTensor


@pai_dataclass
@dataclass
class MyModelParams(ModelBaseParams, ABC):
    ...


@pai_dataclass
@dataclass
class MyModelAParams(MyModelParams):
    def graph_cls(self):
        raise NotImplementedError  # Implement the actual graph

    @staticmethod
    def cls():
        return MyModelA


class MyModelA(ModelBase[MyModelAParams]):
    def _loss(self, inputs, targets, outputs) -> Dict[str, AnyTensor]:
        raise NotImplementedError


@pai_dataclass
@dataclass
class MyModelBParams(MyModelParams):
    def graph_cls(self):
        raise NotImplementedError  # Implement the actual graph

    @staticmethod
    def cls():
        return MyModelB


class MyModelB(ModelBase[MyModelBParams]):
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


# If you want to add a hierarchy to the data, too, make the Generic a TypeVar similar to the model.
# Here, it is fixed to `MyDataParams` and cannot be replaced anymore.
class MyDataBase(DataBase[MyDataParams]):
    def _input_layer_specs(self) -> Dict[str, tf.TensorSpec]:
        raise NotImplementedError

    def _target_layer_specs(self) -> Dict[str, tf.TensorSpec]:
        raise NotImplementedError


# =====================================================
# Scenario definition


# Recreate the TypeVars (Generics) that shall still be Generic, i.e., replaceable, but change their "parent".
TModelParams = TypeVar("TModelParams", bound=MyModelParams)


# Note: Do the same for TDataParams if you want to modify this in sub scenarios.
# Important: The names of the TypeVars must be identical to the ones in `ScenarioBaseParams`


@pai_dataclass
@dataclass
class MyScenarioParams(ScenarioBaseParams[MyDataParams, TModelParams]):
    ...


# Make the scenario params generic
# Important: Similar to above, use the same TypeVar name as in the parent.
TScenarioParams = TypeVar("TScenarioParams", bound=MyScenarioParams)


class MyScenario(ListFileScenario[TScenarioParams]):
    ...


# Create the sub scenarios
# You can only specify the model params since the data params are fixed within `MyScenarioParams`.


@pai_dataclass
@dataclass
class MySubScenario1Params(MyScenarioParams[MyModelAParams]):
    ...


class MySubScenario1(MyScenario[MySubScenario1Params]):
    ...


# Create a second sub scenario
@pai_dataclass
@dataclass
class MySubScenario2Params(MyScenarioParams[MyModelBParams]):
    ...


class MySubScenario2(MyScenario[MySubScenario2Params]):
    ...
