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
import unittest

from tensorflow.keras.backend import clear_session

from test.tutorial.test_tutorial_full import TutorialScenarioTest
from tfaip.data.databaseparams import DataPipelineParams


class ScenarioTest(TutorialScenarioTest):
    @classmethod
    def default_trainer_params(cls):
        p = super().default_trainer_params()
        p.gen.setup.train = DataPipelineParams(limit=10, batch_size=1)
        p.gen.setup.val = DataPipelineParams(limit=10, batch_size=1)
        p.gen.__post_init__()
        p.gen.train_val.dataset = "fashion_mnist"
        p.gen.train_val.force_train = True  # Use always training data ...
        p.gen.train_val.shuffle = False  # ... and dont shuffle the training data
        p.skip_model_load_test = True
        p.random_seed = 1337
        p.force_eager = False
        p.epochs = 5
        p.samples_per_epoch = 10
        return p


class TestEMA(unittest.TestCase):
    def setUp(self) -> None:
        clear_session()

    def tearDown(self) -> None:
        clear_session()

    def test_ema_on_tutorial(self):
        # Train with ema and without ema with same seeds
        # train loss must be equals, but with ema the validation outcomes must be different
        trainer_params = ScenarioTest.default_trainer_params()
        trainer = ScenarioTest.create_trainer(trainer_params)
        first_train_logs = trainer.train()

        clear_session()
        trainer_params.ema_decay = 0.9
        trainer_params.current_epoch = 0

        trainer = ScenarioTest.create_trainer(trainer_params)
        after_train_logs = trainer.train()

        # loss and acc on train must be equal, but lower on val
        self.assertEqual(first_train_logs["keras_loss"], after_train_logs["keras_loss"])
        self.assertEqual(first_train_logs["raw_loss"], after_train_logs["raw_loss"])
        self.assertEqual(first_train_logs["acc"], after_train_logs["acc"])
        self.assertLess(first_train_logs["val_loss"], after_train_logs["val_loss"])
        self.assertGreaterEqual(first_train_logs["val_acc"], after_train_logs["val_acc"])
