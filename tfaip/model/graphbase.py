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
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter

from tfaip import ModelBaseParams
from tfaip.model.layerbase import LayerBase
from tfaip.model.modelbase import ModelBase
from tfaip.model.print_evaluate_layer import PrintEvaluateLayer, PrintEvaluateLayerInput

if TYPE_CHECKING:
    from tfaip.scenario.scenariobase import ScenarioBase

logger = logging.getLogger(__name__)

TMP = TypeVar("TMP", bound=ModelBaseParams)


class TrainingGraph(tf.keras.models.Model):
    """Training-graph wrapper

    Creates metrics, losses, and the training graph.
    Also the print_evaluate_layer is added to "print" outputs during evaluation.
    """

    def __init__(self, scenario: "ScenarioBase", model: ModelBase, graph):
        super().__init__(name="root")
        self.scenario = scenario
        self.model = model
        self.graph = graph
        self.model_metrics = (
            model.metrics
        )  # TODO(christoph): remove hack, add metrics to metrics to that they are know in tensorboard handler

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

    def train_step(self, data):
        """my logic for one training step.

        This method is called by `Model.make_train_function`.

        This method contain1 the mathematical logic for one step of training.
        This typically includes the forward pass, loss calculation, backpropagation,
        and metric updates.

        Configuration details for *how* this logic is run (e.g. `tf.function` and
        `tf.distribute.Strategy` settings), should be left to
        `Model.make_train_function`, which can also be overridden.

        Args:
          data: A tuple of dicts with the input díct as first entry.

        Returns:
          A `dict` containing values that will be passed to
          `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
          values of the `Model`'s metrics are returned. Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.

        """

        if not self.scenario.params.use_input_gradient_regularization:
            """default behaviour"""
            return_metrics = super().train_step(data)
            return return_metrics
        else:
            self.scenario.check_preconditions_for_input_gradient_regularization()
            """
            creates a combined loss based on regular loss and a loss to achieve input gradient regularization
            1. calculate regular loss from inputs (-> prediction) and labels 
            2. calculate loss gradient to inputs (not weights!)
            3. normalize gradient (divide by l2-norm) => direction
            4. add h * direction to inputs => slightly (small h) modified_inputs
            5. calculate loss from modified inputs and labels (batch mean)
            6. calculate input gradient penality term: (loss difference / h) ^ 2  (batch mean)
            7. get combined loss: regular loss + lamda * input penality term
            8. optimization as usual with combined loss
            """
            data = data_adapter.expand_1d(data)
            x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
            h = tf.constant(self.scenario.params.input_gradient_regularization_h)
            h2 = tf.square(h)
            lmbda = tf.constant(self.scenario.params.input_gradient_regularization_lambda)
            logger.info("using input gradient regularization (h=" + str(h) + " lamda=" + str(lmbda))
            inputs_tensor_key = self.scenario.data.input_tensor_key_for_regularization()
            inputs = x[0][inputs_tensor_key]
            with backprop.GradientTape() as outer_tape:
                with backprop.GradientTape(watch_accessed_variables=False) as inner_tape:
                    inner_tape.watch(inputs)
                    y_pred = self(x, training=True)
                    loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
                grads = inner_tape.gradient(loss, inputs)
                norm = tf.expand_dims(
                    tf.sqrt(tf.reduce_sum(tf.pow(grads, 2), axis=tf.range(1, tf.rank(grads)))), axis=1
                )
                masked = tf.greater(norm, 0)
                ones_like_norm = tf.ones_like(norm)
                fixed_norm = tf.where(masked, norm, ones_like_norm)
                gshapesize = tf.size(tf.shape(grads))
                fnshape = tf.shape(fixed_norm)
                fnshapesize = tf.size(fnshape)
                fnshapepadding = tf.ones(tf.subtract(gshapesize, fnshapesize), dtype=tf.int32)
                padded_fnshape = tf.concat([fnshape, fnshapepadding], 0)
                reshapedfixednorm = tf.reshape(norm, padded_fnshape)
                grad_direction = tf.stop_gradient(tf.math.divide(grads, reshapedfixednorm))
                z = tf.math.add(inputs, tf.math.multiply(grad_direction, h))
                z_input_dict = dict(x[0])
                z_input_dict[inputs_tensor_key] = z
                xh = (z_input_dict, x[1], x[2])
                y_pred_h = self(xh, training=True)
                loss_h = self.compiled_loss(y, y_pred_h, sample_weight, regularization_losses=self.losses)
                mean_loss = tf.reduce_mean(loss)
                mean_reg_loss = tf.reduce_mean(tf.math.divide(tf.math.square(tf.math.subtract(loss_h, loss)), h2))
                combined_loss = tf.math.add(mean_loss, tf.math.multiply(mean_reg_loss, lmbda))
            self.optimizer.minimize(combined_loss, self.trainable_variables, tape=outer_tape)
            self.compiled_metrics.update_state(y, y_pred, sample_weight)
            # Collect metrics to return
            return_metrics = {}
            for metric in self.metrics:
                result = metric.result()
                if isinstance(result, dict):
                    return_metrics.update(result)
                else:
                    return_metrics[metric.name] = result
            return return_metrics


class LAVGraph(tf.keras.models.Model):
    """Graph-Wrapper for LAV

    Create metrics and losses but create the prediction graph.
    """

    def __init__(self, model, prediction_model):
        super().__init__(name="root")
        self.model = model
        self.prediction_model = prediction_model

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
    """Wrapper for the actual graph and the model thus inlcuding metrics and losses.

    The RootGraph creates also the `ModelBase` and the `GraphBase` so that everything is wrapped by a `keras.Layer`.
    """

    def get_config(self):
        cfg = super().get_config()
        cfg["params"] = self._params.to_dict()
        cfg["model_kwargs"] = self._model_kwargs
        cfg["graph_kwargs"] = self._graph_kwargs
        return cfg

    @classmethod
    def from_config(cls, config):
        config["params"] = ModelBaseParams.from_dict(config["params"])
        return super().from_config(config)

    def __init__(
        self,
        params: ModelBaseParams,
        name="root",
        setup_graph=True,
        setup_model=True,
        model_kwargs={},
        graph_kwargs={},
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self._model_kwargs = model_kwargs
        self._graph_kwargs = graph_kwargs
        self._params = params
        self._model = self.create_model(**model_kwargs) if setup_model else None
        self._graph: GraphBase = self.create_graph(**graph_kwargs) if setup_graph else None

    def create_model(self, **kwargs):
        return self.params.create(**kwargs)

    def create_graph(self, **kwargs):
        return self.params.create_graph(**kwargs)

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
        """Create the graph.

        Do not call this function manually (`RootGraph(x)`), instead call `train` or `predict`.
        """
        return self._graph(inputs_targets)

    def train(self, inputs, targets):
        """Wrapper for the actual `call` to indicate training."""
        wrap = tf.keras.layers.Lambda(lambda x: {"training": x})
        return self(wrap((inputs, targets)))

    def predict(self, inputs):
        """Wrapper for the actual `call` to indicate training."""
        wrap = tf.keras.layers.Lambda(lambda x: {"predict": x})
        return self(wrap(inputs))

    def pre_proc_targets(self, inputs, targets):
        """Additional function to be called to pre-process the targets within the model.

        The pre-processed targets are then passed to the metrics and losses.

        This is helpful if, e.g., some transformation is know to the graph (e.g. a tokenizer) that is applied on the inputs and targets.
        """
        return self._graph.pre_proc_targets(inputs, targets)


class GenericGraphBase(LayerBase[TMP], ABC):
    """Base class for all user graphs.

    A `GenerticGraphBase` can have a different setup for training and prediction (e.g., S2S-decoding).
    If the training and prediction graphs are identical, inherit `GraphBase` instead.

    Call `train` to create the training graph.
    Call `predict` to create the prediction graph.
    """

    def call(self, inputs_targets, training=None, **kwargs):
        """Create the graph.

        The `inputs_targets` must be a `dict` of either `{'predict': inputs}` or `{'training': inputs_targets}`.
        This call must not be called manually, instead call `train` and `predict`.
        """
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
        """Wrapper for the actual `call` to indicate training."""
        wrap = tf.keras.layers.Lambda(lambda x: {"training": x})
        return self(wrap((inputs, targets)))

    def predict(self, inputs):
        """Wrapper for the actual `call` with indicated prediction."""
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
