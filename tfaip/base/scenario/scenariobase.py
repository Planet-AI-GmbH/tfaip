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
import importlib
import inspect
import json
import os
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import logging
from typing import Type, TYPE_CHECKING, Tuple, List, Optional, Iterable, Dict
import tensorflow as tf
import tensorflow.keras as keras
import re

from tensorflow_addons.optimizers import MovingAverage

from tfaip.base.data.data import DataBase
from tfaip.base.model.exportgraph import ExportGraph
from tfaip.base.scenario.scenariobaseparams import ScenarioBaseParams, NetConfigParamsBase, NetConfigNodeSpec
from tfaip.base.scenario.util.keras_debug_model import KerasDebugModel
from tfaip.base.scenario.util.print_evaluate_layer import PrintEvaluateLayer

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

if TYPE_CHECKING:
    from tfaip.base import TrainerParams
    from tfaip.base.model import ModelBase

logger = logging.getLogger(__name__)


# Label metric because it is used by keras fit as output suffix (e.g. ACC_metric)
def metric(t, p):
    return p


def list_scenarios_in_module(module) -> List[Tuple[str, Type['ScenarioBase']]]:
    return inspect.getmembers(module, lambda member: inspect.isclass(member)
                                                     and member.__module__ == module.__name__
                                                     and issubclass(member, ScenarioBase)
                                                     and member != ScenarioBase
                                                     and "ScenarioBase" not in member.__name__)


@dataclass_json
@dataclass
class KerasModelData:
    loss_names: List[str]
    extended_metric_names: List[str]


class ScenarioBase(ABC):
    """
    The scenario base handles the setup of the scenario including training and model exporting.

    Several abstract methods must be overridden that define how to create the Model, and Data for this scenario.
    If your scenario requires different default params in contrast to the values defined in the respective dataclasses,
    override default_params().
    Furthermore it is possible to change the Trainer or LAV class which is usually not required!

    A typical Scenario only requires to implement get_meta() and get_param_cls()
    To set scenario specific (hidden) parameters use the subclasses __init__ or override create_model() or create_data().
    Note that create_model() will be called after create_data()!

    Attributes:
        data (DataBase):  After setup() was called, the DataBase of the scenario
        model (ModelBase):  After setup() was called, the ModelBase of the scenario

    """

    @classmethod
    def default_params(cls) -> ScenarioBaseParams:
        scenario_params = cls.get_params_cls()()
        scenario_params.scenario_base_path_ = inspect.getfile(cls)
        scenario_params.scenario_module_ = cls.__module__
        scenario_params.model_params = cls.model_cls().get_params_cls()()
        scenario_params.data_params = cls.data_cls().get_params_cls()()
        return scenario_params

    @classmethod
    def params_from_dict(cls, d: dict) -> ScenarioBaseParams:
        params: ScenarioBaseParams = cls.get_params_cls().from_dict(d)
        params.model_params = cls.model_cls().get_params_cls().from_dict(d['model_params'])
        params.data_params = cls.data_cls().get_params_cls().from_dict(d['data_params'])
        return params

    @classmethod
    def from_dict(cls, d: dict) -> Tuple[Type['ScenarioBase'], ScenarioBaseParams]:
        scenario_params = ScenarioBaseParams.from_dict(d)
        spec = importlib.util.spec_from_file_location(scenario_params.scenario_module_,
                                                      scenario_params.scenario_base_path_)
        if not spec:
            raise ModuleNotFoundError(f"Could not find {scenario_params.scenario_base_path_}")

        scenario_module = importlib.util.module_from_spec(spec)
        scenario_cls = None
        try:
            spec.loader.exec_module(scenario_module)
            sys.modules[scenario_params.scenario_module_] = scenario_module
            scenario_name, scenario_cls = list_scenarios_in_module(scenario_module)[0]  # TODO: do not use first: by name instead
        except FileNotFoundError:
            logger.warning(f"Could not load Scenario at '{scenario_params.scenario_base_path_}'")
            from tfaip.scenario import scenarios
            for scenario in scenarios():
                if scenario.name in scenario_params.scenario_module_:
                    logger.info(f"Using scenario '{scenario.name}' instead.")
                    scenario_cls = scenario.scenario

        if scenario_cls is None:
            raise FileNotFoundError(f"Could not load scenario.")

        return scenario_cls, scenario_cls.params_from_dict(d)

    @classmethod
    @abstractmethod
    def data_cls(cls) -> Type['DataBase']:
        raise NotImplemented

    @classmethod
    @abstractmethod
    def model_cls(cls) -> Type['ModelBase']:
        raise NotImplemented

    @classmethod
    def trainer_cls(cls) -> Type['Trainer']:
        from tfaip.base.trainer import Trainer
        return Trainer

    @classmethod
    def lav_cls(cls) -> Type['LAV']:
        from tfaip.base.lav import LAV
        return LAV

    @classmethod
    def create_trainer(cls, trainer_params: 'TrainerParams', restore=False) -> 'Trainer':
        return cls.trainer_cls()(trainer_params, cls(trainer_params.scenario_params), restore)

    @classmethod
    def create_lav(cls, lav_params: 'LAVParams', scenario_params: 'ScenarioBaseParams') -> 'LAV':
        return cls.lav_cls()(lav_params, lambda: cls.data_cls()(scenario_params.data_params),
                             lambda: cls.model_cls()(scenario_params.model_params))

    @staticmethod
    def get_params_cls() -> Type[ScenarioBaseParams]:
        """
        Reference to the actual Scenario Params of this class. Must be overridden by a custom implementation
        :return: ScenatioBaseMeta
        """
        return ScenarioBaseParams

    @staticmethod
    def get_net_config_params_cls() -> Type[NetConfigParamsBase]:
        return NetConfigParamsBase

    def __init__(self, params: ScenarioBaseParams):
        self._params = params

        # Track the global step
        self._keras_model_data = KerasModelData([], [])
        self._keras_train_model: keras.Model = None
        self._export_graphs: Dict[str, ExportGraph] = {}
        self._keras_predict_model: keras.Model = None
        self.data: DataBase = None
        self.model: ModelBase = None

    @property
    def params(self):
        return self._params

    @property
    def keras_train_model(self):
        return self._keras_train_model

    @property
    def keras_predict_model(self):
        return self._keras_predict_model

    def setup(self):
        if not self.data:
            self.data = self.create_data()

        if not self.model:
            self.model = self.create_model()

    def create_data(self) -> 'DataBase':
        return self.__class__.data_cls()(self._params.data_params)

    def create_model(self) -> 'ModelBase':
        return self.__class__.model_cls()(self._params.model_params)

    def print_params(self):
        logger.info("scenario_params=" + self._params.to_json(indent=2))

    def best_logging_settings(self) -> Tuple[str, str]:
        return self.model.best_logging_settings()

    def export(self, path: str, trainer_params: Optional['TrainerParams'] = None, export_resources: bool = True):
        trainer_params_dict = trainer_params.to_dict() if trainer_params else None
        scenario_params_dict = trainer_params_dict['scenario_params'] if trainer_params_dict else self._params.to_dict()
        if export_resources:
            self._export_resources(path, scenario_params_dict)

        # Export frozen model
        full_model_func = tf.function(lambda x: self._keras_predict_model(x))
        full_model_concrete = full_model_func.get_concrete_function(self._keras_predict_model.input)
        # lower_control_flow=True enables tf1 compatibility by disabling tf2 control flow for ops like if/while
        frozen_func = convert_variables_to_constants_v2(full_model_concrete, lower_control_flow=True)
        frozen_func.graph.as_graph_def()
        path_frozen = os.path.join(path, 'frozen')
        os.makedirs(path_frozen, exist_ok=True)
        id_frozen = 'frozen_model.pb'
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                          logdir=path_frozen,
                          name=id_frozen,
                          as_text=False)

        # Export serve models
        for label, export_graph in self._export_graphs.items():
            if label == 'default':
                path_serve = os.path.join(path, 'serve')  # default model handled separately
            else:
                path_serve = os.path.join(path, 'additional', label)
            os.makedirs(path_serve, exist_ok=True)
            export_graph.model.save(path_serve, include_optimizer=False)

        with open(os.path.join(path, 'net_config.json'), 'w') as f:
            json.dump(self.net_config().to_dict(), f, indent=2)

        if trainer_params_dict:
            params_path = os.path.join(path, 'trainer_params.json')
            logger.debug("Storing trainer params to '{}'".format(params_path))
            with open(params_path, 'w') as f:
                json.dump(trainer_params_dict, f, indent=2)
        else:
            params_path = os.path.join(path, 'scenario_params.json')
            with open(params_path, 'w') as f:
                json.dump(scenario_params_dict, f, indent=2)

    def _export_resources(self, root_path: str, scenario_params_dict: dict):
        os.makedirs(os.path.join(root_path), exist_ok=True)
        self.data.dump_resources(root_path, scenario_params_dict['data_params'])

    def _set_no_train_scope(self, regex: Optional[str]):
        if not regex:
            return

        logger.info(f"Setting no train scope to {regex}")
        no_train_scope = re.compile(regex)

        try:
            def _set_for_layer(layer, prefix):
                # This code is for TF 2.3.0 only
                for layer in layer._flatten_layers(False, False):
                    full_layer_name = prefix + '/' + layer.name if prefix else layer.name
                    if no_train_scope.fullmatch(full_layer_name):
                        logger.info(f"Excluding layer '{full_layer_name}' from training")
                        layer.trainable = False
                    else:
                        _set_for_layer(layer, full_layer_name)

            _set_for_layer(self._keras_train_model, '')
        except Exception as e:
            raise Exception("Setting no train scopes was designed for TF 2.3.0. Maybe you use an incompatible version")

    def setup_training(self, optimizer, skip_model_load_test=False, run_eagerly=False,
                       no_train_scope: Optional[str] = None):
        self.setup()

        if self._params.debug_graph_construction:
            if not run_eagerly:
                raise ValueError("Setting debug_graph_construction requires --train_params force_eager=True")
            logger.info("Debugging Graph Construction. Breakpoints during construction are supported")
            # process one example in KerasDebugModel which builds the graph based on this example
            keras_debug_model = KerasDebugModel(self.model)
            with self.data:
                out = keras_debug_model.predict(
                    self._wrapped_train_data(steps_per_epoch=1).take(self._params.debug_graph_n_examples))
            logger.info("Mean values of debug model output: {}".format({k: v.mean() for k, v in out.items()}))

        # This regroups all inputs/targets as input to allow to access the complete data during training
        # This is required to allow for custom loss functions that require multiple inputs (e.g., ctc)
        # This code does:
        # * regroup all inputs/targets as inputs
        # * create loss based on all inputs (includes the gt) and the outputs of the network
        # * the loss itself is handled as "output" of the network, the actual loss function (.compile) just
        #   forwards the "output" loss, this requires to create empty dummy outputs for the loss
        # * see fit()
        #

        real_inputs = self.data.create_input_layers()
        real_targets = self.data.create_target_as_input_layers()

        # all inputs (step and epoch shall have a dimension of [1])
        inputs_targets = {**real_inputs, **real_targets,
                          'step': keras.layers.Input([1], name='step', dtype='int32'),
                          'epoch': keras.layers.Input([1], name='epoch', dtype='int32')}

        # Inputs have already correct names (checked by data) for exporting
        # real_inputs = {k: v if v.op.name == k else keras.layers.Layer(name=k)(v) for k, v in real_inputs.items()}
        # network outputs (ignores the targets)
        logger.info("Building training graph")
        real_outputs = self.model.build(inputs_targets)

        logger.info("Building prediction graph")
        pred_outputs = self.model.build(real_inputs)
        # rename outputs to dict keys (for export)
        pred_outputs = {k: v if v.op.name == k else tf.identity(v, name=k) for k, v in pred_outputs.items()}
        for k, v in pred_outputs.items():
            if v.op.name != k:
                raise NameError(
                    f"Name of output operation {v.op.name} could not be set to {k} in the prediction graph. "
                    f"This usually happens if the same name is used twice in a graph.")

        # inject the evaluate layer to the first output.
        # Note, if the first output is not used in the graph, nothing will be printed
        pel_key = next(iter(real_outputs.keys()))
        logger.debug(f"Injecting print evaluate layer to output {pel_key}")
        if self._params.print_eval_limit != 0:
            pel = PrintEvaluateLayer(self, self._params.print_eval_limit)
            real_outputs[pel_key] = tf.identity(pel((real_outputs[pel_key], real_inputs, real_outputs, real_targets)), name=pel_key + '_')

        # loss as "output" of the network but called separately for logic
        _additional_outputs = self.model.additional_outputs(real_inputs, real_outputs)
        extended_outputs = {**real_outputs, **self.model.additional_outputs(real_inputs, real_outputs)}
        _losses = self.model.loss(inputs_targets, extended_outputs)
        _extended_metrics = self.model.extended_metric(inputs_targets, extended_outputs)
        _simple_metrics = self.model.metric()
        outputs = {
            **real_outputs, **_losses, **_extended_metrics, **_additional_outputs,
            **{k: extended_outputs[v.output] for k, v in _simple_metrics.items()}
        }

        self._keras_model_data = KerasModelData(
            list(_losses.keys()),
            list(_extended_metrics.keys()),
        )

        # create the model (and a second one for exporting)
        logger.info("Building training keras model")
        self._keras_train_model = keras.Model(inputs=inputs_targets, outputs=outputs)
        logger.info("Attempting to set no train scope")
        self._set_no_train_scope(no_train_scope)  # exclude layers from training
        # self._keras_export_model = keras.Model(inputs=real_inputs, outputs=real_outputs)
        logger.info("Building prediction/export keras model (for export and decoding)")
        self._export_graphs = self.model.export_graphs(real_inputs, pred_outputs, real_targets)
        self._keras_predict_model = self._export_graphs['default'].model

        # self._keras_export_model = self._create_export_model()

        # compile the model but with a dummy loss that just returns the 'output' loss
        # the same goes for the metric
        def wrap_loss(_, p):
            return p

        logger.info("Compiling training model including optimization")
        self._keras_train_model.compile(optimizer=optimizer,
                                        loss={k: wrap_loss for k, _ in _losses.items()},
                                        loss_weights=self.model.loss_weights(),
                                        weighted_metrics={**{k: metric for k, _ in _extended_metrics.items()},
                                                          **{k: v.metric for k, v in _simple_metrics.items()}},
                                        run_eagerly=run_eagerly,
                                        )
        logger.info("Compiling prediction model graph")
        self._keras_predict_model.compile(run_eagerly=False)  # Always as graph
        logger.info("Models successfully constructed")

        # check if loading/saving of model works (before actual training)
        if not skip_model_load_test:
            logger.info("Checking if model can be saved")
            with tempfile.TemporaryDirectory() as tmp:
                self._keras_predict_model.save(tmp)
                logger.info("Prediction model successfully saved. Attempting to load it")
                keras.models.load_model(tmp)
            logger.info("Model can be successfully loaded")

    def _wrap_data(self, dataset, steps_per_epoch, batch_size):
        # wrapper for model fit (allows for other arguments)
        def regroup(inputs, targets):
            # see setup_training
            # regroup data for training into all as input, and only loss and metrics as output
            # this is required to allow for custom losses with multiple inputs
            ones_batched = [1] * batch_size
            if self._keras_train_model:
                step_epoch = {'step': ones_batched * self._keras_train_model.optimizer.iterations,
                              'epoch': ones_batched * self._keras_train_model.optimizer.iterations // steps_per_epoch}
            else:
                # No train model exists, this happens on model debug
                step_epoch = {'step': ones_batched * 0,
                              'epoch': ones_batched * 0}
            wrapped_inputs = {**inputs, **targets, **step_epoch}
            wrapped_targets = {**{l: [0] * batch_size for l in
                                  self._keras_model_data.loss_names + self._keras_model_data.extended_metric_names},
                               **{k: targets[v.target] for k, v in self.model.metric().items() if v.target in targets}
                               }
            wrapped_weights = self.model.sample_weights(inputs, targets)
            return wrapped_inputs, wrapped_targets, wrapped_weights

        return dataset.map(regroup)

    def _wrapped_train_data(self, steps_per_epoch):
        return self._wrap_data(self.data.get_train_data(), steps_per_epoch, self._params.data_params.train_batch_size)

    def _wrapped_val_data(self, steps_per_epoch):
        return self._wrap_data(self.data.get_val_data(), steps_per_epoch, self._params.data_params.val_batch_size)

    def fit(self,
            epochs,
            callbacks,
            steps_per_epoch,
            initial_epoch=0,
            **kwargs):
        self._keras_train_model.summary(print_fn=logger.info)

        with self.data:
            self._keras_train_model.fit(self._wrapped_train_data(steps_per_epoch),
                                        epochs=epochs,
                                        callbacks=callbacks,
                                        steps_per_epoch=steps_per_epoch,
                                        initial_epoch=initial_epoch,
                                        validation_data=self._wrapped_val_data(steps_per_epoch),
                                        shuffle=False,
                                        **kwargs
                                        )

        if isinstance(self._keras_train_model.optimizer, MovingAverage):
            # store export model with averaged optimizer weights
            self._keras_train_model.optimizer.assign_average_vars(self._keras_train_model.variables)

    def net_config(self) -> NetConfigParamsBase:
        def parse_spec(nodes, is_input):
            # TODO Evaluate whether the naming of in/out tensors could be inferred somehow...
            # Output naming for frozen can be inferred from 'full_model_concrete.structured_outputs'
            # For keras model no idea yet... even dir(XY) didn't helped
            # However, this 'solution' should work
            res = dict()
            if is_input:
                base_name_frozen = 'x'
                base_name_serve = 'serving_default_'
            else:
                base_name_frozen = 'Identity'
                base_name_serve = 'StatefulPartitionedCall:'

            for idx, (k, v) in enumerate(nodes.items()):
                node_frozen = base_name_frozen
                if idx > 0:
                    node_frozen +='_' + str(idx)
                node_frozen += ':0'
                node_serve = base_name_serve
                if is_input:
                    node_serve += k + ':0'
                else:
                    node_serve += str(idx)

                res[k] = NetConfigNodeSpec(
                    shape=list(map(str, v.shape.dims)) if isinstance(v.shape.dims, Iterable) else [],
                    dtype=v.dtype.name,
                    node_frozen=node_frozen,
                    node_serve=node_serve
                )
            return res

        net_config = self.__class__.get_net_config_params_cls()(
            id_model=self.params.id_,
            id_frozen=os.path.join('frozen', 'frozen_model.pb'),
            id_serve='serve',
            in_nodes=parse_spec(self._keras_predict_model.input, True),
            out_nodes=parse_spec(self._keras_predict_model.output, False),
            tf_version=tf.__version__,
        )

        self._fill_net_config(net_config)
        return net_config

    def _fill_net_config(self, net_config: NetConfigParamsBase):
        pass
