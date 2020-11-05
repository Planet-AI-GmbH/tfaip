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
import json
from abc import ABC
import logging
import os
from datetime import datetime
from typing import Type, Tuple, Union
import tensorflow as tf
from tensorflow.python.keras.callbacks import ProgbarLogger
from typeguard import typechecked

from tfaip.base.device_config import DeviceConfig, distribute_strategy
from tfaip.base.scenario import ScenarioBase
from tfaip.base.trainer.callbacks.benchmark_callback import BenchmarkCallback
from tfaip.base.trainer.callbacks.ema_callback import EMACallback
from tfaip.base.trainer.callbacks.export_best import ExportBestCallback
from tfaip.base.trainer.callbacks.lav_callback import LAVCallback
from tfaip.base.trainer.callbacks.logger_callback import LoggerCallback
from tfaip.base.trainer.callbacks.tensor_board_callback import TensorBoardCallback
from tfaip.base.trainer.callbacks.tensorflow_fix import TensorflowFix
from tfaip.base.trainer.callbacks.train_params_logger import TrainParamsLoggerCallback
from tfaip.base.trainer.callbacks.fix_metric_labels import FixMetricLabelsCallback
from tfaip.base.trainer.optimizer.gradient_accumulation_optimizer import create_gradient_accumulation_optimizer
from tfaip.base.trainer.trainerparams import TrainerParams
from tfaip.base.trainer.warmstart.warmstarter import Warmstarter
from tfaip.base.trainer.optimizer.weights_moving_average import WeightsMovingAverage
from tfaip.util.random import set_global_random_seed

logger = logging.getLogger(__name__)


class Trainer(ABC):
    """
    The Trainer class is typically identical for all scenarios. Its purpose is to set up the training callbacks,
    Warmstarting/Restarting. The training loop is wrapped in ScenarioBase and is a call to keras.Model.fit.
    """

    @staticmethod
    def get_params_cls() -> Type[TrainerParams]:
        return TrainerParams

    @staticmethod
    def restore_trainer(checkpoint: Union[str, dict]) -> 'Trainer':
        if isinstance(checkpoint, str):
            with open(os.path.join(checkpoint, 'trainer_params.json')) as f:
                d = json.load(f)
        else:
            d = checkpoint

        trainer_params: TrainerParams = TrainerParams.from_dict(d)
        logger.info("trainer_params=" + trainer_params.to_json(indent=2))
        scenario, scenario_params = ScenarioBase.from_dict(d['scenario_params'])

        # Load the actual scenario params for the particular scenario
        trainer_params.scenario_params = scenario_params
        trainer = scenario.create_trainer(trainer_params, restore=True)
        return trainer

    @typechecked
    def __init__(self, params: TrainerParams, scenario: ScenarioBase, restore=False):
        super(Trainer, self).__init__()
        self._params = params
        self.restore = restore
        if self._params.random_seed is not None:
            set_global_random_seed(self._params.random_seed)

        if restore and not self._params.checkpoint_dir:
            raise ValueError("To restore a training, a checkpoint dir must be provided")

        self.device_config = DeviceConfig(self._params.device_params)

        # default value of export best shall be true if a checkpoint dir is provided
        # if the user manually sets it to true, a checkpoint dir must be provided
        if params.export_best is None:
            params.export_best = params.checkpoint_dir is not None
        if params.export_best and not params.checkpoint_dir:
            raise ValueError("To use 'export_best' a 'checkpoint_dir' must be specified")
        if self._params.checkpoint_dir:
            scenario.params.id_ = os.path.basename(self._params.checkpoint_dir) + "_"
        else:
            scenario.params.id_ = ''
        scenario.params.id_ = scenario.params.id_ + scenario.params.scenario_module_ + "_" + datetime.today().strftime('%Y-%m-%d')

        if self._params.force_eager:
            tf.config.run_functions_eagerly(run_eagerly=True)

        self._warmstarter = Warmstarter(self._params.warmstart_params)
        self._scenario = scenario
        scenario.setup()
        self._data = scenario.data
        self._model = scenario.model
        self.stop_training = False

        self._steps_per_epoch = self._params.samples_per_epoch // self._data.params().train_batch_size
        if self._steps_per_epoch <= 0:
            raise ValueError(f"Samples per epoch must be greater than the train batch size, but got "
                             f"{self._params.samples_per_epoch} < {self._data.params().train_batch_size}")

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(self._params.tf_cpp_min_log_level)

    @property
    def scenario(self):
        return self._scenario

    @property
    def params(self):
        return self._params

    @distribute_strategy
    def train(self, callbacks=None):
        external_callbacks = callbacks
        callbacks = []

        self._params.learning_rate_params.epochs_ = self._params.epochs
        self._params.learning_rate_params.steps_per_epoch_ = self._steps_per_epoch
        optimizer = self._create_optimizer()

        self._scenario.setup_training(optimizer, self._params.skip_model_load_test,
                                      run_eagerly=self._params.force_eager,
                                      no_train_scope=self._params.no_train_scope)
        if self.restore:
            logger.info("Restoring from checkpoint '{}'".format(self._params.checkpoint_dir))
            # load_weights also restores the optimizer weights!
            self._scenario.keras_train_model.load_weights(
                os.path.join(self._params.checkpoint_dir, 'variables', 'variables'))
            if self._params.warmstart_params.model:
                logger.warning("Ignoring warmstart since training is resumed from a checkpoint")
        else:
            self._warmstarter.warmstart(self._scenario.keras_train_model)

        callbacks.append(FixMetricLabelsCallback())
        callbacks.append(ProgbarLogger(count_mode='steps'))  # Progbar after label fix
        callbacks.append(TensorflowFix())
        callbacks.append(BenchmarkCallback())

        if self._params.lav_every_n >= 1:
            # LAV callback must be assigned before export best to allow to export based on best LAV
            callbacks.append(LAVCallback(self._params, self._scenario))

        if self._params.checkpoint_dir:
            if self._params.write_checkpoints:
                # we need only weights to restore the training, the graph will be reconstructed
                variables_path = os.path.join(self._params.checkpoint_dir, 'variables', 'variables')
                callbacks.append(tf.keras.callbacks.ModelCheckpoint(variables_path, save_weights_only=True))
            callbacks.append(TrainParamsLoggerCallback(self._params, self._params.checkpoint_dir))

        if self._params.calc_ema:
            # EMA must be before export best to export ema
            # noinspection PyTypeChecker
            callbacks.append(EMACallback(optimizer))

        if self._params.export_best and self._params.checkpoint_dir:
            callbacks.append(
                ExportBestCallback(os.path.join(self._params.checkpoint_dir, 'best'), self._scenario, self._params))

        if self._params.checkpoint_dir:
            # Tensorflow Callback as last, so that it is allowed to add additional outputs (e.g. LAVCallback)
            callbacks.append(TensorBoardCallback(log_dir=self._params.checkpoint_dir,
                                                 steps_per_epoch=self._steps_per_epoch,
                                                 reset=self._params.current_epoch == 0,
                                                 profile="10,20" if self._params.profile else 0))

        callbacks.append(LoggerCallback())
        if external_callbacks:
            callbacks.extend(external_callbacks)

        if self._params.epochs <= self._params.current_epoch:
            logger.warning(
                f"Attempting to train until epoch {self._params.current_epoch} but the model was already trained for "
                f"{self._params.current_epoch} epochs. Final export only.")
        else:
            logger.info(
                f"Starting training in epoch {self._params.current_epoch}. "
                f"{self._params.epochs - self._params.current_epoch} remaining.")
            if self._params.random_seed is not None:
                # Set fixed random seed for training if desired, this makes training independent of previous operations
                # such as loading/creating model from scratch
                set_global_random_seed(self._params.random_seed + 1)
            self._scenario.fit(epochs=self._params.epochs,
                               initial_epoch=self._params.current_epoch,
                               steps_per_epoch=self._steps_per_epoch,
                               callbacks=callbacks,
                               )

        # export the model to "checkpoint_dir/export"
        if self._params.checkpoint_dir and self._params.export_final:
            logger.info("Final export of the model.")
            self._scenario.export(os.path.join(self._params.checkpoint_dir, 'export'))

    @typechecked
    def _create_optimizer(self) -> tf.keras.optimizers.Optimizer:
        # Create the optimizer
        # Wrap with calc_ema if desired
        @typechecked
        def optimizer_class() -> Tuple[Type[tf.keras.optimizers.Optimizer], dict]:
            # returns the optimizer (either the real one, or wrapped with calc ema)
            # do not return actual instance since gradient accumulation_optimizer will override the given optimizer
            real_optimizer = getattr(tf.keras.optimizers, self._params.optimizer_params.optimizer)
            clip_grad = self._params.optimizer_params.clip_grad
            args = {
                "learning_rate": self._params.learning_rate_params.create(),
                "clipnorm": clip_grad if clip_grad > 0 else None,
                "clipvalue": -clip_grad if clip_grad < 0 else None,
            }
            if self._params.calc_ema:
                return WeightsMovingAverage, {'optimizer': real_optimizer(**args)}
            else:
                return real_optimizer, args

        # create the gradient accumulation optimizer (will not wrap, if train_accum_steps <= 1)
        return create_gradient_accumulation_optimizer(self._params.train_accum_steps, *optimizer_class())
