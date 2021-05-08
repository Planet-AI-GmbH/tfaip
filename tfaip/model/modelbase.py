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
from abc import abstractmethod, ABC
from typing import Type, Dict, Any, Tuple, Optional, List, TYPE_CHECKING, TypeVar, Generic

import tensorflow as tf
from typeguard import typechecked

from tfaip import ModelBaseParams
from tfaip import Sample
from tfaip.data.data import DataBase
from tfaip.model.losses.definitions import LossDefinition
from tfaip.model.metric.definitions import MetricDefinition
from tfaip.model.metric.multi import MultiMetricDefinition
from tfaip.util.tftyping import AnyTensor

if TYPE_CHECKING:
    from tfaip.model.graphbase import GraphBase
    from tfaip.trainer.callbacks.tensor_board_data_handler import TensorBoardDataHandler

logger = logging.getLogger(__name__)

TMP = TypeVar('TMP', bound=ModelBaseParams)


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
    def all_custom_objects(cls) -> Dict[str, Any]:
        general_layers = {}
        for c in cls._additional_layers():
            name = c.__name__
            if name in general_layers:
                logger.warning(f'Class names must be unique, but class with name "{name}". Consider to rename it!')
            general_layers[name] = c

        return general_layers

    @classmethod
    @typechecked
    def additional_layers(cls) -> List[Type[tf.keras.layers.Layer]]:
        """
        List all custom layers of the model. This is required to enable eager mode in LAV.
        (See e.g. Tutorial for an example)

        Returns:
            List of all layers
        """
        return cls._additional_layers()

    @classmethod
    def _additional_layers(cls) -> List[Type[tf.keras.layers.Layer]]:
        from tfaip.model.util.module import import_graphs  # pylint: disable=import-outside-toplevel
        try:
            return import_graphs(cls.__module__)
        except ModuleNotFoundError:
            logger.error('Could not find additional layers automatically. Either create a graphs.py file or graphs '
                         'package (directory) where GraphBase implementations are searched, or override '
                         '_additional_layers in your Model')
            raise

    def __init__(self, params: TMP, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._params: TMP = params
        self._graph = None
        self._tensorboard_handler = self._create_tensorboard_handler()

    def setup(self):
        if not self._graph:
            self._graph = self.create_graph(self._params)

    @property
    def params(self) -> TMP:
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
        # Override this function
        return 'min', 'loss'

    @typechecked
    def build(self, inputs_targets: Dict[str, AnyTensor]) -> Dict[str, AnyTensor]:
        """
        Override _build for custom implementation. Do this with caution
        :param inputs_targets: Dictionary of both the inputs and the targets
        :return: The outputs of the model
        """
        self.setup()
        return self._graph(inputs_targets)

    @abstractmethod
    def create_graph(self, params: TMP) -> 'GraphBase':
        raise NotImplementedError

    @typechecked()
    def additional_outputs(self, inputs: Dict[str, AnyTensor], outputs: Dict[str, AnyTensor]) -> Dict[str, AnyTensor]:
        return self._additional_outputs(inputs, outputs)

    def _additional_outputs(self, inputs: Dict[str, tf.Tensor], outputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        # Override this function
        del inputs  # Not required in the default implementation
        del outputs  # Not required in the default implementation
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
        # wrap the metric into a layer, this is required since the outputs of _extended_metric can be simple Tensors
        # but everything pure tensorflow backend call must be wrapped into a layer
        return _LambdaLayerNoConfig(self._extended_metric)((inputs_targets, outputs))

    def _extended_metric(self, inputs_targets: Dict[str, tf.Tensor], outputs: Dict[str, tf.Tensor]
                         ) -> Dict[str, tf.Tensor]:
        # Override this function
        del inputs_targets  # Not required in the default implementation
        del outputs  # Not required in the default implementation
        return {}

    @typechecked
    def metric(self) -> Dict[str, MetricDefinition]:
        """Override _metric in a custom implementation. Standard metrics allow for one input and one target only, and also
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
        # Override this function
        return {}

    def _multi_metric(self) -> Dict[str, MultiMetricDefinition]:
        # Override this function
        return {}

    @typechecked
    def sample_weights(self, inputs: Dict[str, AnyTensor], targets: Dict[str, AnyTensor]) -> Dict[str, Any]:
        """The weights of the samples. The output key must match the respective metric, extended_metric, or loss name.
        Thus, if you compute the loss "CTC" and the metric "CER" and "CAR" and all three shall be weighted, return a
        dictionary with three entries but the same values.

        :param inputs:  The inputs of the model
        :param targets:   The outputs of the model
        :return: Dictionary of the weights
        """
        return self._sample_weights(inputs, targets)

    def _sample_weights(self, inputs: Dict[str, tf.Tensor], targets: Dict[str, tf.Tensor]) -> Dict[str, Any]:
        # Override this function
        del inputs  # Not required in the default implementation
        del targets  # Not required in the default implementation
        return {}

    @typechecked
    def extended_loss(self, inputs_targets: Dict[str, AnyTensor], outputs: Dict[str, AnyTensor]
                      ) -> Dict[str, AnyTensor]:
        """
        A dictionary of all losses of the model that will be averaged if there are multiple.
        Only override _extended_loss for the custom implementation

        :param inputs_targets:  Inputs and targets of the model
        :param outputs:  Outputs of the model
        :return:  Dictionary of the loss
        """
        # wrap the loss into a layer, this is required since the outputs of _extended_loss can be simple Tensors
        # but everything pure tensorflow backend call must be wrapped into a layer
        return _LambdaLayerNoConfig(self._extended_loss)((inputs_targets, outputs))

    def _extended_loss(self, inputs_targets: Dict[str, tf.Tensor], outputs: Dict[str, tf.Tensor]
                       ) -> Dict[str, AnyTensor]:
        """
        Override to implement a loss as a Tensor output

        See Also:
            _loss
        """
        del inputs_targets  # Not required in the default implementation
        del outputs  # Not required in the default implementation
        return {}

    @typechecked
    def loss(self) -> Dict[str, LossDefinition]:
        """
        Losses based on keras.losses. Implement _loss
        Returns:
            A dict of the loss name and a LossDefinition
        See Also:
            extended_loss
        """
        losses = self._loss()
        if 'loss' in losses:
            name = 'loss_loss'
            while name in losses:
                name = name + '_loss'
            logger.warning('Cannot use "loss" as loss name because it is a reserved name by keras.'
                           f'Automatically renaming to "{name}" but you should consider to rename it.')
            losses[name] = losses['loss']
            del losses['loss']

        for k, loss in losses.items():
            loss.loss.name = k
        return losses

    def _loss(self) -> Dict[str, LossDefinition]:
        # Implement this
        return {}

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
    def export_graphs(self,
                      inputs: Dict[str, AnyTensor],
                      outputs: Dict[str, AnyTensor],
                      targets: Dict[str, AnyTensor],
                      ) -> Dict[str, tf.keras.Model]:
        eg = self._export_graphs(inputs, outputs, targets)
        if 'default' not in eg:
            raise KeyError(f'Expected at least an export graph with label "default" in {eg}.')
        return eg

    def _export_graphs(self,
                       inputs: Dict[str, tf.Tensor],
                       outputs: Dict[str, tf.Tensor],
                       targets: Dict[str, tf.Tensor],
                       ) -> Dict[str, tf.keras.Model]:
        # Override this function
        del targets  # not required in the default implementation
        return {"default": tf.keras.Model(inputs=inputs, outputs=outputs)}

    @property
    def tensorboard_handler(self):
        return self._tensorboard_handler

    def _create_tensorboard_handler(self) -> 'TensorBoardDataHandler':
        # Override this function
        from tfaip.trainer.callbacks.tensor_board_data_handler import \
            TensorBoardDataHandler  # pylint: disable=import-outside-toplevel
        return TensorBoardDataHandler()


class _LambdaLayerNoConfig(tf.keras.layers.Layer):
    """Implementation of a lambda layer that can be serialized by keras

    The class can however not be deserialized. So only use it during training.
    """

    def get_config(self):
        p = super().get_config()
        p['fn'] = None  # this layer cannot be instantiated
        return p

    def __init__(self, fn, **kwargs):
        super().__init__(**kwargs)
        self.fn = fn
        if fn is None:
            raise ValueError('Parameter model is None. If this error occurs during loading a model, it means that '
                             'the Training Graph was loaded or this layer was used in the Prediction graph. '
                             'This is not supported.')

    def call(self, *args, **kwargs):
        del kwargs  # not used here
        return self.fn(*args[0])
