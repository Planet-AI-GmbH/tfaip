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
from typing import Type, Dict, Any, Tuple, Optional, List, TYPE_CHECKING
import tensorflow as tf
from typeguard import typechecked
import logging

from tfaip.base.data.data import DataBase
from tfaip.base.model.exportgraph import ExportGraph
from tfaip.base.model.metric.multi import MultiMetricDefinition
from tfaip.base.model.metric.simple import MetricDefinition
from tfaip.base.model.modelbaseparams import ModelBaseParams
from tfaip.util.typing import AnyNumpy, AnyTensor

if TYPE_CHECKING:
    from tfaip.base.model import GraphBase
    from tfaip.base.trainer.callbacks.tensor_board_data_handler import TensorBoardDataHandler

logger = logging.getLogger(__name__)


class ModelBase(ABC):
    """
    The ModelBase class provides the implementation of the keras Model, its losses and metrics.

    You must inherit get_params_cls() to provide the actual dataclass for the ModelParams.
    Only override the private methods _loss, _metrics, _extended_metrics to allow for type checking at runtime.
    The _build method will construct your graph.
    """

    @staticmethod
    @abstractmethod
    def get_params_cls() -> Type[ModelBaseParams]:
        raise NotImplementedError

    @classmethod
    def get_all_custom_objects(cls) -> Dict[str, Any]:
        general_layers = {}
        for c in cls._get_additional_layers():
            name = c.__name__
            if name in general_layers:
                logger.warning(f"Class names must be unique, but class with name {name}. "
                               f"Consider to rename it!")
            general_layers[name] = c

        return general_layers

    @classmethod
    @typechecked
    def get_additional_layers(cls) -> List[Type[tf.keras.layers.Layer]]:
        """
        List all custom layers of the model. This is required to enable eager mode in LAV. (See e.g. Tutorial for an example)
        :return: List of all Layers
        """
        return cls._get_additional_layers()

    @classmethod
    def _get_additional_layers(cls) -> List[Type[tf.keras.layers.Layer]]:
        from tfaip.base.model.util.module import import_graphs
        return import_graphs(cls.__module__)

    def __init__(self, params: ModelBaseParams, *args, **kwargs):
        super(ModelBase, self).__init__(*args, **kwargs)
        self._params = params
        self._graph = None
        self._tensorboard_handler: TensorBoardDataHandler = self._create_tensorboard_handler()

    def params(self) -> ModelBaseParams:
        return self._params

    @typechecked
    def best_logging_settings(self) -> Tuple[str, str]:
        """
        Which metric/loss shall be logged, and if the minimum or maximum of this value is better. E. G.:
        "min", "CER" or "max", "ACC" or "min", "loss"
        The metric must match the name of the logger
        :return: str, str
        """
        return self._best_logging_settings()

    def _best_logging_settings(self) -> Tuple[str, str]:
        return "min", "loss"

    @typechecked
    def build(self, inputs_targets: Dict[str, AnyTensor]) -> Dict[str, AnyTensor]:
        """
        Override _build for custom implementation. Do this with caution
        :param inputs_targets: Dictionary of both the inputs and the targets
        :return: The outputs of the model
        """
        if not self._graph:
            self._graph = self.create_graph(self._params)
        return self._graph(inputs_targets)

    @abstractmethod
    def create_graph(self, params: ModelBaseParams) -> 'GraphBase':
        raise NotImplementedError

    @typechecked()
    def additional_outputs(self, inputs: Dict[str, AnyTensor], outputs: Dict[str, AnyTensor]) -> Dict[str, AnyTensor]:
        return self._additional_outputs(inputs, outputs)

    def _additional_outputs(self, inputs: Dict[str, tf.Tensor], outputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        return {}

    @typechecked
    def extended_metric(self,
                        inputs_targets: Dict[str, AnyTensor],
                        outputs: Dict[str, AnyTensor]
                        ) -> Dict[str, AnyTensor]:
        """
        use lambda layers, you can not use self.<variables> directly, it will result in pickle-error
        Override _extended_metric for custom implementation.
        :param inputs_targets: A dictionary containing both the inputs and the targets of the model
        :param outputs: A dictionary providing the outputs of the graph
        :return: A dictionary of metric values
        """
        return self._extended_metric(inputs_targets, outputs)

    def _extended_metric(self, inputs: Dict[str, tf.Tensor], outputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        return {}

    @typechecked
    def metric(self) -> Dict[str, MetricDefinition]:
        """
        note: targets - holds input and output dict? eager execution is not working here, since it is
                'map'ped on the tf.dataset
        Override _metric in a custom implementation. Standard metrics allow for one input and one target only, and also
        have access to the sample weights.

        :return: A Dictionary of MetricDefinition
        """
        metrics = self._metric()
        # convert multi metrics to simple metrics
        for k, v in self._multi_metric().items():
            for c in v.metric.children:
                metrics[c.name] = MetricDefinition(v.target, v.output, c)
            metrics[k] = MetricDefinition(v.target, v.output, v.metric)

        return metrics

    def _metric(self) -> Dict[str, MetricDefinition]:
        return {}

    def _multi_metric(self) -> Dict[str, MultiMetricDefinition]:
        return {}

    @typechecked
    def sample_weights(self, inputs: Dict[str, AnyTensor], targets: Dict[str, AnyTensor]) -> Dict[str, Any]:
        """
        note: targets - holds input and output dict? eager execution is not working here, since it is
                'map'ped on the tf.dataset
        The weights of the samples. The output key must match the respective metric, extended_metric, or loss name.
        Thus, if you compute the loss "CTC" and the metric "CER" and "CAR" and all three shall be weighted, return a
        dictionary with three entries but the same values.

        :param inputs:  The inputs of the model
        :param targets:   The outputs of the model
        :return: Dictionay of the weights
        """
        return self._sample_weights(inputs, targets)

    def _sample_weights(self, inputs: Dict[str, tf.Tensor], targets: Dict[str, tf.Tensor]) -> Dict[str, Any]:
        return {}

    @typechecked
    def loss(self, inputs_targets: Dict[str, AnyTensor], outputs: Dict[str, AnyTensor]) -> Dict[str, AnyTensor]:
        """
        A dictionary of all losses of the model that will be averaged if there are multiple.
        Only override _loss for the custom implementation

        :param inputs_targets:  Inputs and targets of the model
        :param outputs:  Outputs of the model
        :return:  Dictionary of the loss
        """
        return self._loss(inputs_targets, outputs)

    @abstractmethod
    def _loss(self, inputs: Dict[str, tf.Tensor], outputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """use lambda layers, you can not use self.<variables> directly, it will result in pickle-error"""
        raise NotImplementedError

    @typechecked
    def loss_weights(self) -> Optional[Dict[str, float]]:
        """
        An optional dictionary of the weights of the losses. Override _loss_weights for custom implementation.
        :return: loss name and its weight
        """
        return self._loss_weights()

    def _loss_weights(self) -> Optional[Dict[str, float]]:
        pass

    @typechecked
    def print_evaluate(self, inputs: Dict[str, AnyNumpy], outputs: Dict[str, AnyNumpy], targets: Dict[str, AnyNumpy],
                       data: DataBase, print_fn=print):
        """
        Print evaluation output
        :param inputs: Inputs of the model
        :param outputs: Outputs of the model
        :param targets: Targets of the model
        :param data: The data class of the scenario
        :param print_fn:  the print function to use
        """
        self._print_evaluate(inputs, outputs, targets, data, print_fn)

    def _print_evaluate(self, inputs: Dict[str, AnyNumpy], outputs: Dict[str, AnyNumpy], targets: Dict[str, AnyNumpy],
                        data: DataBase, print_fn):
        # Default implementation that should be overwritten by the actual model
        target, prediction = self.target_prediction(targets, outputs, data)
        print_fn(f"\n    TARGET: {target}\nPREDICTION: {prediction}")

    @typechecked
    def target_prediction(self,
                          targets: Dict[str, AnyNumpy],
                          outputs: Dict[str, AnyNumpy],
                          data: DataBase
                          ) -> Tuple[Any, Any]:
        t, p = self._target_prediction(targets, outputs, data)
        """
        Return the actual final target and prediction (e.g. the strings in ATR).
        The output can be dumped during lav for further analysis, e.g. comparing of different models
        :except
        :param outputs: Outputs of the model
        :param targets: Targets of the model
        :param data: The data class of the scenario
        """
        if type(t) != type(p):
            raise TypeError(f"Prediction and target must be the same type but got {type(p)} and {type(t)} with values"
                            f"{p} and {t}")

        return t, p

    def _target_prediction(self,
                           targets: Dict[str, AnyNumpy],
                           outputs: Dict[str, AnyNumpy],
                           data: DataBase,
                           ) -> Tuple[Any, Any]:
        return None, None

    @typechecked()
    def export_graphs(self,
                      inputs: Dict[str, AnyTensor],
                      outputs: Dict[str, AnyTensor],
                      targets: Dict[str, AnyTensor],
                      ) -> Dict[str, ExportGraph]:
        eg = {g.label: g for g in self._export_graphs(inputs, outputs, targets)}
        if 'default' not in eg:
            raise KeyError(f"Expected at least an export graph with label 'default' in {eg}.")
        return eg

    def _export_graphs(self,
                       inputs: Dict[str, tf.Tensor],
                       outputs: Dict[str, tf.Tensor],
                       targets: Dict[str, tf.Tensor],
                       ) -> List[ExportGraph]:
        return [ExportGraph("default", inputs=inputs, outputs=outputs)]


    @property
    def tensorboard_handler(self):
        return self._tensorboard_handler

    def _create_tensorboard_handler(self) -> 'TensorBoardDataHandler':
        from tfaip.base.trainer.callbacks.tensor_board_data_handler import TensorBoardDataHandler
        return TensorBoardDataHandler()
