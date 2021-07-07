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
"""Definition of the Trainer"""
import json
import logging
import os
from abc import ABC
from datetime import datetime
from typing import Type, Tuple, Union, TypeVar, Generic, Optional, Dict

import tensorflow as tf
from tfaip import TrainerParams
from tfaip.device.device_config import DeviceConfig, distribute_strategy
from tfaip.scenario.scenariobase import ScenarioBase
from tfaip.trainer.callbacks.benchmark_callback import BenchmarkCallback
from tfaip.trainer.callbacks.earlystopping.callback import EarlyStoppingCallback
from tfaip.trainer.callbacks.ema_callback import EMACallback
from tfaip.trainer.callbacks.extract_logs import ExtractLogsCallback
from tfaip.trainer.callbacks.lav_callback import LAVCallback
from tfaip.trainer.callbacks.logger_callback import LoggerCallback
from tfaip.trainer.callbacks.progbar import TFAIPProgbarLogger
from tfaip.trainer.callbacks.tensor_board_callback import TensorBoardCallback
from tfaip.trainer.callbacks.tensorflow_fix import TensorflowFix
from tfaip.trainer.callbacks.train_params_logger import TrainerCheckpointsCallback
from tfaip.trainer.optimizer.gradient_accumulation_optimizer import create_gradient_accumulation_optimizer
from tfaip.trainer.optimizer.weights_moving_average import WeightsMovingAverage
from tfaip.trainer.scheduler.learningrate import LearningRateSchedule
from tfaip.trainer.scheduler.schedule_weightdecay import WeightDecaySchedule
from tfaip.trainer.warmstart.warmstarter import WarmStarter
from tfaip.util.generic_meta import CollectGenericTypes
from tfaip.util.random import set_global_random_seed
from tfaip.util.typing import AnyNumpy
from typeguard import typechecked

logger = logging.getLogger(__name__)

TTrainerParams = TypeVar("TTrainerParams", bound=TrainerParams)


class Trainer(Generic[TTrainerParams], ABC, metaclass=CollectGenericTypes):
    """
    The Trainer class is typically identical for all scenarios. Its purpose is to set up the training callbacks,
    Warmstarting/Restarting. The training loop is wrapped in ScenarioBase and is a call to keras.Model.fit.
    """

    @classmethod
    def params_cls(cls) -> Type[TTrainerParams]:
        return cls.__generic_types__[TTrainerParams.__name__]

    @staticmethod
    def parse_trainer_params(d: Union[str, dict]) -> Tuple[TTrainerParams, Type[ScenarioBase]]:
        if isinstance(d, str):
            if not d.endswith(".json"):
                d = os.path.join(d, "trainer_params.json")

            with open(d) as f:
                d = json.load(f)
        scenario, scenario_params = ScenarioBase.from_dict(d["scenario"])
        trainer_params: TrainerParams = scenario.trainer_cls().params_cls().from_dict(d)
        logger.info(f"trainer_params={trainer_params.to_json(indent=2)}")

        # Load the actual scenario params for the particular scenario
        trainer_params.scenario = scenario_params
        return trainer_params, scenario

    @classmethod
    def restore_trainer(cls, checkpoint: Union[str, dict]) -> "Trainer":
        trainer_params, scenario = cls.parse_trainer_params(checkpoint)
        logger.info(f"trainer_params={trainer_params.to_json(indent=2)}")
        trainer = scenario.create_trainer(trainer_params, restore=True)
        return trainer

    @typechecked
    def __init__(self, params: TTrainerParams, scenario: ScenarioBase, restore=False):
        super().__init__()
        self._params = params
        self.restore = restore
        if self._params.random_seed is not None:
            set_global_random_seed(self._params.random_seed)

        if restore and not self._params.output_dir:
            raise ValueError("To restore a training, a checkpoint dir must be provided")

        self.device_config = DeviceConfig(self._params.device)

        # default value of export best shall be true if a checkpoint dir is provided
        # if the user manually sets it to true, a checkpoint dir must be provided
        if params.export_best is None:
            params.export_best = params.output_dir is not None
        if params.export_best and not params.output_dir:
            raise ValueError("To use 'export_best' a 'output_dir' must be specified")
        if self._params.output_dir:
            scenario.params.id = os.path.basename(self._params.output_dir) + "_"
        else:
            scenario.params.id = ""
        scenario.params.id = (
            scenario.params.id + scenario.params.scenario_id + "_" + datetime.today().strftime("%Y-%m-%d")
        )

        self._scenario = scenario
        self.stop_training = False
        self._steps_per_epoch: Optional[int] = None  # Not initialized yet
        self._callbacks = []
        self._data = None
        self._model = None

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(self._params.tf_cpp_min_log_level)

    @property
    def scenario(self):
        return self._scenario

    @property
    def params(self):
        return self._params

    def setup_data(self):
        if self._data is not None:
            return

        self._data = self._scenario.data

    def setup_model(self):
        if self._model is not None:
            return

        if self._params.random_seed is not None:
            # Set fixed random seed for training if desired, this makes training independent of previous operations
            # such as loading/creating model from scratch
            set_global_random_seed(self._params.random_seed + 1)

        self._model = self._scenario.model

    @distribute_strategy
    def train(self, callbacks=None) -> Dict[str, AnyNumpy]:
        """Start training

        Returns:
            The last logs
        """
        self.setup_data()
        self.setup_model()
        self.setup_steps_per_epoch()

        self._params.learning_rate.epochs = self._params.epochs
        self._params.learning_rate.steps_per_epoch = self._steps_per_epoch
        optimizer = self._create_optimizer()

        # create pipelines (so that they can be accessed in scenario by mode)
        self._data.get_or_create_pipeline(self.params.gen.setup.train, self.params.gen.train_gen())
        self._data.get_or_create_pipeline(self.params.gen.setup.val, self.params.gen.val_gen())

        self._scenario.setup_training(
            optimizer,
            self._params.skip_model_load_test,
            run_eagerly=self._params.force_eager,
            no_train_scope=self._params.no_train_scope,
        )
        if self.restore:
            logger.info(f"Restoring from checkpoint '{self._params.output_dir}'")
            # load_weights also restores the optimizer weights!
            self._scenario.keras_train_model.load_weights(
                os.path.join(self._params.output_dir, self._params.saved_checkpoint_sub_dir, "variables", "variables")
            )
            if self._params.warmstart.model:
                logger.warning("Ignoring warmstart since training is resumed from a checkpoint")
        else:
            # Use the "predict" model here since it only comprises relevant weights (no metrics, etc.)
            custom_objects = self._model.all_custom_objects()
            self.create_warmstarter().warmstart(self._scenario.keras_predict_model, custom_objects)

        callbacks = self.setup_callbacks(optimizer, callbacks)
        logger_callback = next(c for c in callbacks if isinstance(c, LoggerCallback))  # get the logger callback

        if self._params.epochs <= self._params.current_epoch:
            logger.warning(
                f"Attempting to train until epoch {self._params.current_epoch} but the model was already trained for "
                f"{self._params.current_epoch} epochs. Final export only."
            )
        else:
            logger.info(
                f"Starting training in epoch {self._params.current_epoch}. "
                f"{self._params.epochs - self._params.current_epoch} remaining."
            )

            self._callbacks = callbacks
            self.fit()

        # export the model to "output_dir/export"
        if self._params.output_dir and self._params.export_final:
            logger.info("Final export of the model.")
            self._scenario.export(os.path.join(self._params.output_dir, "export"))

        return logger_callback.last_logs

    def create_train_params_logger_callback(self, store_weights, store_params):
        if self._params.output_dir:
            save_freq = self._params.checkpoint_save_freq
            if self._params.write_checkpoints:
                if isinstance(save_freq, str) and save_freq.isdigit():
                    save_freq = int(save_freq)
                if isinstance(save_freq, int):
                    save_freq = save_freq * self._steps_per_epoch
                if save_freq == 0:
                    save_freq = None
        else:
            save_freq = None

        return TrainerCheckpointsCallback(
            self._params, save_freq, store_weights=store_weights, store_params=store_params
        )

    def setup_callbacks(
        self,
        optimizer,
        callbacks=None,
    ):
        external_callbacks = callbacks
        callbacks = []

        extract_logs_cb = ExtractLogsCallback()
        callbacks.append(extract_logs_cb)
        callbacks.append(TFAIPProgbarLogger(delta_time=self._params.progbar_delta_time, count_mode="steps"))
        callbacks.append(TensorflowFix())
        callbacks.append(BenchmarkCallback(extract_logs_cb))
        # split storing of parameters and weights
        # first store the actual checkpoint (non EMA weights)
        # after the EarlyStoppingCallback which must be listed after EMACallback we can store the updated trainer
        # params since the EarlyStoppingCallback will change the params
        callbacks.append(self.create_train_params_logger_callback(store_params=False, store_weights=True))

        if self._params.ema_decay != 0.0:
            # EMA must be before export best to export ema
            # noinspection PyTypeChecker
            callbacks.append(EMACallback(optimizer))

        if self._params.lav_every_n >= 1:
            # LAV callback depends on EMACallback
            # LAV callback must be assigned before export best to allow to export based on best LAV
            callbacks.append(LAVCallback(self._params, self._scenario, extract_logs_cb))

        # EarlyStoppingCallback depends on LAVCallback and EMA Callback
        callbacks.append(EarlyStoppingCallback(self._scenario, self._params))

        # Now we can store the params, depends on EarlyStoppingCallback
        callbacks.append(self.create_train_params_logger_callback(store_params=True, store_weights=False))

        if self._params.output_dir:
            # Tensorflow Callback as last, so that it is allowed to add additional outputs (e.g. LAVCallback)
            callbacks.append(
                TensorBoardCallback(
                    log_dir=self._params.output_dir,
                    steps_per_epoch=self._steps_per_epoch,
                    extracted_logs_cb=extract_logs_cb,
                    reset=self._params.current_epoch == 0,
                    profile="10,20" if self._params.profile else 0,
                )
            )

        callbacks.append(LoggerCallback())
        if external_callbacks:
            callbacks.extend(external_callbacks)

        return callbacks

    def setup_steps_per_epoch(self):
        if self._params.samples_per_epoch < 0:
            logger.info(
                f"Setting samples per epoch relative to dataset size with a factor of "
                f"{self._params.scale_epoch_size}. Note that this "
                "requires the creation of the data generator once before training."
            )
            samples_per_epoch = len(self._params.gen.train_data(self._data).create_data_generator())

            if self._params.scale_epoch_size != 1:
                samples_per_epoch = int(samples_per_epoch * self._params.scale_epoch_size)

            if samples_per_epoch <= 0:
                raise ValueError(
                    "Could not compute the number of samples per epoch based on the size of the data "
                    "generator. Please implement __len__ correctly."
                )
            logger.info(f"Set samples per epoch to {samples_per_epoch}")
        else:
            samples_per_epoch = self._params.samples_per_epoch
            if self._params.scale_epoch_size != 1:
                logger.warning(
                    "Setting scale_epoch_size has no effect when using absolute values for samples_per_epoch."
                    "Set samples_per_epoch to the default (=-1) to use relative computation."
                )

        self._steps_per_epoch = samples_per_epoch // self.params.gen.setup.train.batch_size
        if self._steps_per_epoch <= 0:
            raise ValueError(
                f"Samples per epoch must be greater than the train batch size, but got "
                f"{samples_per_epoch} < {self.params.gen.setup.train.batch_size}"
            )

    def fit(self):
        self._scenario.fit(
            epochs=self._params.epochs,
            initial_epoch=self._params.current_epoch,
            steps_per_epoch=self._steps_per_epoch,
            validation_freq=self._params.val_every_n,
            callbacks=self._callbacks,
            verbose=self._params.progress_bar_mode,
        )

    @typechecked
    def _create_optimizer(self) -> tf.keras.optimizers.Optimizer:
        # Create the optimizer
        # Wrap with ema_decay if desired
        @typechecked
        def optimizer_class() -> Tuple[Type[tf.keras.optimizers.Optimizer], dict]:
            # returns the optimizer (either the real one, or wrapped with calc ema)
            # do not return actual instance since gradient accumulation_optimizer will override the given optimizer
            real_optimizer, args = self._params.optimizer.create()
            lr_schedule = self._params.learning_rate.create()
            args["learning_rate"] = lr_schedule
            if "weight_decay" in args:
                if isinstance(lr_schedule, LearningRateSchedule):
                    args["weight_decay"] = WeightDecaySchedule(args["weight_decay"], lr_schedule)

            if self._params.ema_decay != 0.0:
                if self._params.ema_decay >= 1:
                    raise ValueError(
                        f"The EMA decay is {self._params.ema_decay} >= 1 which is invalid. Either pass "
                        f"a negative value for an automatic computation, or a value in (0, 1)."
                    )
                elif self._params.ema_decay < 0.0:
                    emadecay = 0.75 ** (
                        float(self._params.gen.setup.train.batch_size) / max(self._params.samples_per_epoch, 1)
                    )
                    # Very short epochs lead to low ema decays. prevent this...
                    emadecay = max(emadecay, 0.99)
                    # Very long epochs lead to very very high ema decays. prevent this...
                    emadecay = min(emadecay, 0.99975)
                else:
                    emadecay = self._params.ema_decay
                return WeightsMovingAverage, {"optimizer": real_optimizer(**args), "average_decay": emadecay}
            else:
                return real_optimizer, args

        # create the gradient accumulation optimizer (will not wrap, if train_accum_steps <= 1)
        return create_gradient_accumulation_optimizer(self._params.train_accum_steps, *optimizer_class())

    def create_warmstarter(self) -> WarmStarter:
        return WarmStarter(self.params.warmstart)
