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
"""Implementation of the ModelBase"""
import logging
from abc import ABC, abstractmethod
from typing import Type, Dict, Any, Tuple, Optional, List, TYPE_CHECKING, TypeVar, Generic

import tensorflow as tf
from typeguard import typechecked

from tfaip import ModelBaseParams
from tfaip import Sample
from tfaip.data.data import DataBase
from tfaip.model.metric.count import Count
from tfaip.model.metric.multi import MultiMetricDefinition
from tfaip.model.tensorboardwriter import TensorboardWriter
from tfaip.util.tftyping import AnyTensor

if TYPE_CHECKING:
    from tfaip.model.graphbase import RootGraph

logger = logging.getLogger(__name__)

TMP = TypeVar("TMP", bound=ModelBaseParams)


class ModelBase(Generic[TMP], ABC):
    """
    The ModelBase class provides the implementation of the keras Model, its losses and metrics.
    """

    @classmethod
    def params_cls(cls) -> Type[TMP]:
        arg = cls.__orig_bases__[0].__args__[0]
        if isinstance(arg, TypeVar):
            return arg.__bound__  # default
        return arg

    @classmethod
    def all_custom_objects(cls) -> Dict[str, Type[tf.keras.layers.Layer]]:
        """Custom objects required to instantiate saved keras models in eager mode (reinstantiation)"""
        root_graph = cls.root_graph_cls()
        return {
            root_graph.__name__: root_graph,
            **cls.base_custom_objects(),
        }

    @classmethod
    def base_custom_objects(cls) -> Dict[str, Type[tf.keras.layers.Layer]]:
        """Custom objects required to instantiate saved keras models even in graph mode"""
        return {
            "TensorboardWriter": TensorboardWriter,
        }

    def __init__(self, params: TMP, **kwargs):
        self._params: TMP = params
        self._count_metric = Count()

    @staticmethod
    def root_graph_cls() -> Type["RootGraph"]:
        from tfaip.model.graphbase import RootGraph

        return RootGraph

    @property
    def params(self) -> TMP:
        return self._params

    @typechecked
    def best_logging_settings(self) -> Tuple[str, str]:
        """
        Which metric/loss shall be logged, and if the minimum or maximum of this value is better. E. G.:
        "min", "CER" or "max", "ACC" or "min", "loss/mean_epoch"
        The metric must match the name of the logger
        :return: str, str
        """
        return self._best_logging_settings()

    def _best_logging_settings(self) -> Tuple[str, str]:
        # Override this function
        return "min", "loss/mean_epoch"

    @typechecked
    def metric(self, inputs, targets, outputs) -> List[AnyTensor]:
        """The metrics of the model

        Override _metric in a custom implementation.

        Instantiate keras metrics in a Models init function and return the called metric here.
        """
        metrics = self._metric(inputs, targets, outputs)
        return metrics

    def _target_output_metric(self) -> List[Tuple[str, str, tf.keras.metrics.Metric]]:
        """Deprecated"""
        return []

    def _metric(self, inputs, targets, outputs) -> List[AnyTensor]:
        # Override this function
        return []

    def _multi_metric(self) -> List[MultiMetricDefinition]:
        # Override this function
        return []

    @typechecked
    def sample_weights(self, inputs: Dict[str, AnyTensor], targets: Dict[str, AnyTensor]) -> Dict[str, Any]:
        """The weights of the samples. The output key must match the respective metric, extended_metric, or loss name.
        Thus, if you compute the loss "CTC" and the metric "CER" and "CAR" and all three shall be weighted, return a
        dictionary with three entries but the same values.

        :param inputs:  The inputs of the model
        :param targets:   The outputs of the model
        :return: Dictionary of the weights
        """
        logger.warning(
            "Sample weights are deprecated and should not be called anymore."
            "Sample weights are still required for deprecated multi metrics, though."
        )
        return self._sample_weights(inputs, targets)

    def _sample_weights(self, inputs: Dict[str, tf.Tensor], targets: Dict[str, tf.Tensor]) -> Dict[str, Any]:
        # Override this function
        del inputs  # Not required in the default implementation
        del targets  # Not required in the default implementation
        return {}

    @typechecked
    def loss(self, inputs, targets, outputs) -> Dict[str, AnyTensor]:
        """Returns a list of losses

        The losses will be collected by weighting with the loss_weights that default to 1
        """
        return self._loss(inputs, targets, outputs)

    @abstractmethod
    def _loss(self, inputs, targets, outputs) -> Dict[str, AnyTensor]:
        # Implement this
        raise NotImplementedError

    @typechecked
    def loss_weights(self) -> Optional[Dict[str, float]]:
        """
        An optional dictionary of the weights of the losses. Override _loss_weights for custom implementation.
        :return: loss name and its weight
        """
        return self._loss_weights()

    def _loss_weights(self) -> Optional[Dict[str, float]]:
        # Override this function
        pass

    @typechecked
    def print_evaluate(self, sample: Sample, data: DataBase, print_fn=print):
        """
        Print evaluation output
        :param sample: an unbatched sample
        :param data: The data class of the scenario
        :param print_fn:  the print function to use
        """
        self._print_evaluate(sample, data, print_fn)

    def _print_evaluate(self, sample: Sample, data: DataBase, print_fn):
        # Override this function
        # Default implementation that should be overwritten by the actual model
        pass

    @typechecked()
    def export_graphs(
        self,
        inputs: Dict[str, AnyTensor],
        outputs: Dict[str, AnyTensor],
        targets: Dict[str, AnyTensor],
    ) -> Dict[str, tf.keras.Model]:
        eg = self._export_graphs(inputs, outputs, targets)
        if "default" not in eg:
            raise KeyError(f'Expected at least an export graph with label "default" in {eg}.')
        return eg

    def _export_graphs(
        self,
        inputs: Dict[str, tf.Tensor],
        outputs: Dict[str, tf.Tensor],
        targets: Dict[str, tf.Tensor],
    ) -> Dict[str, tf.keras.Model]:
        # Override this function
        del targets  # not required in the default implementation
        return {"default": tf.keras.Model(inputs=inputs, outputs=outputs)}

    def add_all_losses(self, model, inputs, targets, outputs):
        loss_weights = self.loss_weights() or {}
        total_loss = 0
        for name, loss in self.loss(inputs, targets, outputs).items():
            loss_weight = loss_weights.get(name, 1)
            loss_v = tf.reduce_mean(loss)
            extra_loss = loss_v * loss_weight
            total_loss += extra_loss
            model.add_metric(loss_v, name=name)

        model.add_loss(total_loss)
        model.add_metric(total_loss, name="loss/mean_epoch")

    def add_all_metrics(self, model, inputs, targets, outputs):
        for metric_v in self.metric(inputs, targets, outputs):
            model.add_metric(metric_v)

        # add counter metric
        model.add_metric(self._count_metric(inputs, targets))

        # convert multi metrics to simple metrics
        if len(model.multi_metrics) > 0:
            sample_weights = self.sample_weights(inputs, targets)
            for v in model.multi_metrics:
                for c in v.metric.children:
                    model.add_metric(c(targets[v.target], outputs[v.output], sample_weights.get(c.name, None)))
                model.add_metric(
                    v.metric(targets[v.target], outputs[v.output], sample_weights.get(v.metric.name, None))
                )

        if len(model.target_output_metrics) > 0:
            sample_weights = self.sample_weights(inputs, targets)
            for t, o, m in model.target_output_metrics:
                model.add_metric(m(targets[t], outputs[o], sample_weights.get(m.name, None)))

    def pre_proc_targets(self, inputs, targets):
        return targets

    def post_proc_targets(self, inputs, targets, outputs):
        return targets

    def wrap_model_with_loss_and_metric(self, model, inputs, targets, outputs, with_losses=True, with_metrics=True):
        post_proc_targets = self.post_proc_targets(inputs, targets, outputs)
        if with_losses:
            self.add_all_losses(model, inputs, post_proc_targets, outputs)
        if with_metrics:
            self.add_all_metrics(model, inputs, post_proc_targets, outputs)


class TFAIPKerasModel(tf.keras.models.Model):
    def call(self, inputs, training=None, mask=None):
        return super().call(inputs, training=training, mask=mask)

    def get_config(self):
        return super().get_config()

    def __init__(self, inputs, outputs, **kwargs):
        # Only allow functional API
        super().__init__(inputs, outputs, **kwargs)
