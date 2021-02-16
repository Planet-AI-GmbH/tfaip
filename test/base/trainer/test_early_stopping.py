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

from tensorflow.python.keras.backend import clear_session

from tfaip.base.trainer import TrainerParams
from tfaip.scenario.tutorial.data import DataParams
from tfaip.scenario.tutorial.scenario import TutorialScenario
import logging
logging.basicConfig(level=logging.DEBUG)


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


class TestEarlyStopping(unittest.TestCase):
    def setUp(self) -> None:
        clear_session()

    def tearDown(self) -> None:
        clear_session()

    def test_early_stopping_frequency(self):
        scenario_params = get_default_scenario_params()
        trainer_params = TrainerParams(
            epochs=10,
            samples_per_epoch=10,
            scenario_params=scenario_params,
            skip_model_load_test=True,
            random_seed=1337,
        )
        trainer_params.early_stopping_params.n_to_go = 3
        trainer_params.early_stopping_params.frequency = 2
        trainer_params.learning_rate_params.lr = 0.0  # No updates, so end at epoch 2
        trainer = TutorialScenario.create_trainer(trainer_params)
        trainer.train()

        self.assertEqual(trainer_params.early_stopping_params.frequency * (trainer_params.early_stopping_params.n_to_go - 1),
                         trainer_params.current_epoch,
                         )

    def test_early_stopping_limit(self):
        scenario_params = get_default_scenario_params()
        trainer_params = TrainerParams(
            epochs=10,
            samples_per_epoch=10,
            scenario_params=scenario_params,
            skip_model_load_test=True,
            random_seed=1337,
        )
        trainer_params.early_stopping_params.upper_threshold = 0.5
        trainer = TutorialScenario.create_trainer(trainer_params)
        trainer.train()

        self.assertEqual(trainer_params.current_epoch, 5)

    def test_early_stopping_n_max(self):
        scenario_params = get_default_scenario_params()
        trainer_params = TrainerParams(
            epochs=10,
            samples_per_epoch=10,
            scenario_params=scenario_params,
            skip_model_load_test=True,
            random_seed=1337,
        )
        trainer_params.early_stopping_params.n_to_go = 4
        trainer_params.learning_rate_params.lr = 0.0  # No updates, so end at epoch 2
        trainer = TutorialScenario.create_trainer(trainer_params)
        trainer.train()

        self.assertEqual(trainer_params.current_epoch, trainer_params.early_stopping_params.n_to_go)

    def test_early_stopping_n_max_lower_limit_not_reached(self):
        scenario_params = get_default_scenario_params()
        trainer_params = TrainerParams(
            epochs=5,
            samples_per_epoch=10,
            scenario_params=scenario_params,
            skip_model_load_test=True,
            random_seed=1337,
        )
        trainer_params.early_stopping_params.n_to_go = 2
        trainer_params.early_stopping_params.lower_threshold = 0.5  # this threshold must at least be reached (impossible in this case)
        trainer_params.early_stopping_params.upper_threshold = 0.9  # this wont be reached
        trainer_params.learning_rate_params.lr = 0.0  # No updates, so end at epoch 2
        trainer = TutorialScenario.create_trainer(trainer_params)
        trainer.train()

        self.assertEqual(trainer_params.current_epoch, trainer_params.epochs)

    def test_early_stopping_n_max_lower_limit_reached(self):
        scenario_params = get_default_scenario_params()
        trainer_params = TrainerParams(
            epochs=5,
            samples_per_epoch=10,
            scenario_params=scenario_params,
            skip_model_load_test=True,
            random_seed=1337,
        )
        trainer_params.early_stopping_params.n_to_go = 2
        trainer_params.early_stopping_params.upper_threshold = 0.1  # this threshold must at least be reached (fulfilled in this case)
        trainer_params.early_stopping_params.upper_threshold = 0.9  # this wont be reached
        trainer_params.learning_rate_params.lr = 0.0  # No updates, so end at epoch 2
        trainer = TutorialScenario.create_trainer(trainer_params)
        trainer.train()

        self.assertEqual(trainer_params.current_epoch, trainer_params.early_stopping_params.n_to_go)
