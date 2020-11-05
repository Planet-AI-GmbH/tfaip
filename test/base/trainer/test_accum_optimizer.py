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
import unittest

from tensorflow.keras.backend import clear_session

from test.util.store_logs_callback import StoreLogsCallback
from tfaip.base.trainer import TrainerParams
from tfaip.scenario.tutorial.data import DataParams
from tfaip.scenario.tutorial.scenario import TutorialScenario


def get_default_data_params():
    return DataParams(
        train_batch_size=1,
        val_batch_size=1,
        val_limit=10,
    )


def get_default_scenario_params():
    default_params = TutorialScenario.default_params()
    default_params.data_params = get_default_data_params()
    return default_params


class TestTrainAccumulationOptimizer(unittest.TestCase):
    def setUp(self) -> None:
        clear_session()

    def test_accumulation_on_tutorial_with_ema(self):
        # Train with accum and without accum with same seeds
        # train loss must be equals
        accum = 10
        store_logs_callback = StoreLogsCallback()
        scenario_params = get_default_scenario_params()
        scenario_params.data_params.train_batch_size = accum
        trainer_params = TrainerParams(
            epochs=10,
            samples_per_epoch=accum,
            scenario_params=scenario_params,
            skip_model_load_test=True,
            random_seed=1337,
            train_accum_steps=1,
            force_eager=False,
        )
        trainer = TutorialScenario.create_trainer(trainer_params)
        trainer.train(callbacks=[store_logs_callback])
        first_train_logs = store_logs_callback.logs

        clear_session()
        scenario_params.data_params.train_batch_size = 1
        trainer_params.train_accum_steps = accum
        trainer_params.current_epoch = 0
        trainer_params.calc_ema = True

        store_logs_callback = StoreLogsCallback()
        trainer = TutorialScenario.create_trainer(trainer_params)
        trainer.train(callbacks=[store_logs_callback])

        # loss and acc on train must be equal, but lower on val
        self.assertAlmostEqual(first_train_logs['loss_loss'], store_logs_callback.logs['loss_loss'], places=2,
                               msg='loss_loss')
        self.assertAlmostEqual(first_train_logs['acc_metric'], store_logs_callback.logs['acc_metric'], places=2,
                               msg='acc_metric')
        self.assertLess(first_train_logs['val_loss'], store_logs_callback.logs['val_loss'], "val_loss")
        self.assertGreater(first_train_logs['val_acc_metric'], store_logs_callback.logs['val_acc_metric'],
                           "val_acc_metric")

        clear_session()

    def test_accumulation_on_tutorial(self):
        # Train with accum and without accum with same seeds
        # train loss must be equals
        accum = 3
        store_logs_callback = StoreLogsCallback()
        scenario_params = get_default_scenario_params()
        scenario_params.data_params.train_batch_size = accum
        trainer_params = TrainerParams(
            epochs=3,
            samples_per_epoch=accum,
            scenario_params=scenario_params,
            skip_model_load_test=True,
            random_seed=1337,
            train_accum_steps=1,
            force_eager=False,
        )
        trainer = TutorialScenario.create_trainer(trainer_params)
        trainer.train(callbacks=[store_logs_callback])
        first_train_logs = store_logs_callback.logs

        clear_session()
        scenario_params.data_params.train_batch_size = 1
        trainer_params.train_accum_steps = accum
        trainer_params.current_epoch = 0

        store_logs_callback = StoreLogsCallback()
        trainer = TutorialScenario.create_trainer(trainer_params)
        trainer.train(callbacks=[store_logs_callback])

        for k, v in store_logs_callback.logs.items():
            self.assertAlmostEqual(v, first_train_logs[k], places=6)
        clear_session()
