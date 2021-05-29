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
import logging
import unittest

from tensorflow.keras.backend import clear_session

from test.tutorial.test_tutorial_full import TutorialScenarioTest
from tfaip.data.databaseparams import DataPipelineParams

logging.basicConfig(level=logging.DEBUG)


class ScenarioTest(TutorialScenarioTest):
    @classmethod
    def default_trainer_params(cls):
        p = super().default_trainer_params()
        p.gen.setup.train = DataPipelineParams(batch_size=1)
        p.gen.setup.val = DataPipelineParams(limit=10, batch_size=1)
        p.gen.__post_init__()
        p.gen.train_val.dataset = "fashion_mnist"
        p.skip_model_load_test = True
        p.random_seed = 1337
        p.force_eager = False
        p.epochs = 10
        p.samples_per_epoch = 10
        return p


class TestEarlyStopping(unittest.TestCase):
    def tearDown(self) -> None:
        clear_session()

    def test_early_stopping_frequency(self):
        trainer_params = ScenarioTest.default_trainer_params()
        trainer_params.early_stopping.n_to_go = 3
        trainer_params.early_stopping.frequency = 2
        trainer_params.learning_rate.lr = 0.0  # No updates, so end at epoch 2
        trainer = ScenarioTest.create_trainer(trainer_params)
        trainer.train()

        self.assertEqual(
            trainer_params.early_stopping.frequency * (trainer_params.early_stopping.n_to_go - 1),
            trainer_params.current_epoch,
        )

    def test_early_stopping_limit(self):
        trainer_params = ScenarioTest.default_trainer_params()
        trainer_params.early_stopping.upper_threshold = 0.5
        trainer = ScenarioTest.create_trainer(trainer_params)
        trainer.train()

        self.assertLess(trainer_params.current_epoch, trainer_params.epochs)

    def test_early_stopping_n_max(self):
        trainer_params = ScenarioTest.default_trainer_params()
        trainer_params.early_stopping.n_to_go = 4
        trainer_params.learning_rate.lr = 0.0  # No updates, so end at epoch 2
        trainer = ScenarioTest.create_trainer(trainer_params)
        trainer.train()

        self.assertEqual(trainer_params.current_epoch, trainer_params.early_stopping.n_to_go)

    def test_early_stopping_n_max_lower_limit_not_reached(self):
        trainer_params = ScenarioTest.default_trainer_params()
        trainer_params.epochs = 5
        trainer_params.early_stopping.n_to_go = 2
        trainer_params.early_stopping.lower_threshold = (
            0.5  # this threshold must at least be reached (impossible in this case)
        )
        trainer_params.early_stopping.upper_threshold = 0.9  # this wont be reached
        trainer_params.learning_rate.lr = 0.0  # No updates, so end at epoch 2
        trainer = ScenarioTest.create_trainer(trainer_params)
        trainer.train()

        self.assertEqual(trainer_params.current_epoch, trainer_params.epochs)

    def test_early_stopping_n_max_lower_limit_reached(self):
        trainer_params = ScenarioTest.default_trainer_params()
        trainer_params.epochs = 5
        trainer_params.early_stopping.n_to_go = 2
        trainer_params.early_stopping.upper_threshold = (
            0.1  # this threshold must at least be reached (fulfilled in this case)
        )
        trainer_params.early_stopping.upper_threshold = 0.9  # this wont be reached
        trainer_params.learning_rate.lr = 0.0  # No updates, so end at epoch 2
        trainer = ScenarioTest.create_trainer(trainer_params)
        trainer.train()

        self.assertEqual(trainer_params.current_epoch, trainer_params.early_stopping.n_to_go)
