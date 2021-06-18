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
"""Definition of GraphBase"""
import logging
from abc import ABC, abstractmethod
from typing import TypeVar, TYPE_CHECKING

import tensorflow as tf

from tfaip import ModelBaseParams
from tfaip.model.layerbase import LayerBase
from tfaip.model.modelbase import ModelBase
from tfaip.model.print_evaluate_layer import PrintEvaluateLayer, PrintEvaluateLayerInput

if TYPE_CHECKING:
    from tfaip.scenario.scenariobase import ScenarioBase

logger = logging.getLogger(__name__)

TMP = TypeVar("TMP", bound=ModelBaseParams)


class TrainingGraph(tf.keras.models.Model):
    def __init__(self, scenario: "ScenarioBase", model: ModelBase, graph):
        super().__init__(name="root")
        self.model = model
        self.graph = graph
        self.multi_metrics = model._multi_metric()
        if len(self.multi_metrics) > 0:
            logger.warning(
                "Usage of ModelBase._multi_metric is deprecated. It is now possible to compute everything directly in "
                "the metric function."
            )

        self.target_output_metrics = model._target_output_metric()
        if len(self.target_output_metrics) > 0:
            logger.warning(
                "Usage of ModelBase._target_output_metric (former MetricDefinition) is deprecated. "
                "Create metrics in the init function and call them in metric."
            )

        if scenario.params.print_eval_limit != 0:
            self.print_eval_layer = PrintEvaluateLayer(scenario, scenario.params.print_eval_limit)
            self.print_eval_layer.build({})  # call dummy build to setup variable
        else:
            self.print_eval_layer = None

    def call(self, inputs, training=None, mask=None):
        inputs, targets, meta = inputs
        pre_proc_targets = self.model.pre_proc_targets(inputs, targets)
        outputs = self.graph.train(inputs, pre_proc_targets)
        self.model.wrap_model_with_loss_and_metric(self, inputs, pre_proc_targets, outputs)
        if self.print_eval_layer is not None:
            # Add as dummy loss (non visible, no contribution since pel returns 0)
            self.add_loss(self.print_eval_layer(PrintEvaluateLayerInput(inputs, outputs, pre_proc_targets, meta)))
        return outputs


class LAVGraph(tf.keras.models.Model):
    def __init__(self, model, prediction_model):
        super().__init__(name="root")
        self.model = model
        self.prediction_model = prediction_model
        self.multi_metrics = model._multi_metric()
        if len(self.multi_metrics) > 0:
            logger.warning(
                "Usage of MultiMetric is deprecated. It is now possible to compute everything directly in "
                "the metric function."
            )

        self.target_output_metrics = model._target_output_metric()
        if len(self.target_output_metrics) > 0:
            logger.warning(
                "Usage of ModelBase._target_output_metric (former MetricDefinition) is deprecated. "
                "Create metrics in the init function and call them in metric."
            )

    def call(self, inputs, training=None, mask=None):
        if "lav" not in inputs:
            raise KeyError("Missing lav in inputs. Call graph.lav(inputs) instead of graph(inputs).")

        inputs, targets = inputs["lav"]
        targets = self.model.pre_proc_targets(inputs, targets)
        outputs = self.prediction_model(inputs)
        self.model.wrap_model_with_loss_and_metric(self, inputs, targets, outputs, with_losses=False)
        return outputs, targets

    def lav(self, inputs, targets):
        return self({"lav": (inputs, targets)})


def create_training_graph(scenario: "ScenarioBase", model, graph: "RootGraph") -> tf.keras.models.Model:
    return TrainingGraph(scenario, model, graph)


def create_lav_graph(model, keras_model: tf.keras.models.Model):
    return LAVGraph(model, keras_model)


class RootGraph(tf.keras.layers.Layer):
    def get_config(self):
        cfg = super().get_config()
        cfg["params"] = self._params.to_dict()
        return cfg

    @classmethod
    def from_config(cls, config):
        config["params"] = ModelBaseParams.from_dict(config["params"])
        return super().from_config(config)

    def __init__(self, params: ModelBaseParams, name="root", **kwargs):
        super().__init__(name=name, **kwargs)
        self._params = params
        self._model = self.create_model()
        self._graph: GraphBase = self.create_graph()

    def create_model(self):
        return self.params.create()

    def create_graph(self):
        return self.params.create_graph()

    @property
    def params(self) -> TMP:
        return self._params

    @property
    def model(self):
        return self._model

    @property
    def graph(self) -> "GraphBase":
        return self._graph

    def call(self, inputs_targets, training=None, **kwargs):
        return self._graph(inputs_targets)

    def train(self, inputs, targets):
        wrap = tf.keras.layers.Lambda(lambda x: {"training": x})
        return self(wrap((inputs, targets)))

    def predict(self, inputs):
        wrap = tf.keras.layers.Lambda(lambda x: {"predict": x})
        return self(wrap(inputs))

    def pre_proc_targets(self, inputs, targets):
        return self._graph.pre_proc_targets(inputs, targets)


class GenericGraphBase(LayerBase[TMP], ABC):
    def call(self, inputs_targets, training=None, **kwargs):
        # parse inputs
        if isinstance(inputs_targets, dict) and ("predict" in inputs_targets or "training" in inputs_targets):
            if "predict" in inputs_targets:
                inputs = inputs_targets["predict"]
                outputs = self.build_prediction_graph(inputs)
            else:
                inputs, targets = inputs_targets["training"]
                outputs = self.build_train_graph(inputs, targets, training=training)
        else:
            raise ValueError(
                f"A Graph must not be called directly e.g. by graph(inputs). "
                f"Instead call graph.train(inputs, targets) or graph.predict(inputs) to either construct "
                f"the training or prediction graph (which is however identical in most cases. "
                f"REASON: Got wrong input type {type(inputs_targets)} with value {inputs_targets}."
            )

        return outputs

    def train(self, inputs, targets):
        wrap = tf.keras.layers.Lambda(lambda x: {"training": x})
        return self(wrap((inputs, targets)))

    def predict(self, inputs):
        wrap = tf.keras.layers.Lambda(lambda x: {"predict": x})
        return self(wrap(inputs))

    @abstractmethod
    def build_train_graph(self, inputs, targets=None, training=None):
        raise NotImplementedError

    def build_prediction_graph(self, inputs):
        return self.build_train_graph(inputs)


class GraphBase(GenericGraphBase[TMP], ABC):
    def build_train_graph(self, inputs, targets=None, training=None):
        return self.build_graph(inputs, training)

    @abstractmethod
    def build_graph(self, inputs, training=None):
        raise NotImplementedError
