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
from typing import Dict, Type, Any

import tensorflow as tf
from paiargparse import pai_dataclass

from tfaip import (
    ModelBaseParams,
    DataBaseParams,
    ScenarioBaseParams,
)
from tfaip.data.data import DataBase
from tfaip.model.graphbase import GraphBase
from tfaip.model.modelbase import ModelBase
from tfaip.scenario.listfile.listfilescenario import ListFileScenario
from tfaip.util.tftyping import AnyTensor


# =====================================================
# Model and Graph definition


@pai_dataclass
@dataclass
class MyModelParams(ModelBaseParams):
    def graph_cls(self):
        return MyGraph

    @staticmethod
    def cls():
        return MyModel


class MyModel(ModelBase[MyModelParams]):
    def __init__(self, additional_parameter, **kwargs):
        super().__init__(**kwargs)
        self.additional_parameter = additional_parameter

    def _loss(self, inputs, targets, outputs) -> Dict[str, AnyTensor]:
        raise NotImplementedError


class MyGraph(GraphBase[MyModelParams]):
    def __init__(self, another_parameter, **kwargs):
        super().__init__(**kwargs)
        self.another_parameter = another_parameter

    def build_graph(self, inputs, training=None):
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
    def __init__(self, additional_parameter, **kwargs):
        super().__init__(**kwargs)
        self.additional_parameter = additional_parameter

    def _input_layer_specs(self) -> Dict[str, tf.TensorSpec]:
        raise NotImplementedError

    def _target_layer_specs(self) -> Dict[str, tf.TensorSpec]:
        raise NotImplementedError


@pai_dataclass
@dataclass
class MyScenarioParams(ScenarioBaseParams[MyDataParams, MyModelParams]):
    ...


class MyScenario(ListFileScenario[MyScenarioParams]):
    @staticmethod
    def additional_model_kwargs(data: MyDataBase, scenario_params: MyScenarioParams) -> Dict[str, Any]:
        # Access any parameter of DataBase, which is initialized already, or from the MyScenarioParams
        return {"additional_parameter": data.additional_parameter}

    @staticmethod
    def additional_graph_kwargs(data: MyDataBase, scenario_params: MyScenarioParams) -> Dict[str, Any]:
        # Access any parameter of DataBase, which is initialized already, or from the MyScenarioParams
        return {"another_parameter": data.additional_parameter}

    @staticmethod
    def static_data_kwargs(scenario_params: MyScenarioParams) -> Dict[str, Any]:
        # Data allows only parameters from the full set of MyScenarioParams
        return {"additional_parameter": scenario_params.scenario_id}
