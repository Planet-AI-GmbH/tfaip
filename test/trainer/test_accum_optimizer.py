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
        p.gen.setup.train = DataPipelineParams(batch_size=1, num_processes=1)
        p.gen.setup.val = DataPipelineParams(limit=10, batch_size=1, num_processes=1)
        p.gen.__post_init__()
        p.gen.train_val.shuffle = False  # Do not shuffle so that the same elements are produced
        p.gen.train_val.dataset = "fashion_mnist"
        p.gen.train_val.force_train = True  # Use Train as val
        p.skip_model_load_test = True
        p.random_seed = 1337
        p.force_eager = False
        return p


class TestTrainAccumulationOptimizer(unittest.TestCase):
    def setUp(self) -> None:
        clear_session()

    def tearDown(self) -> None:
        clear_session()

    def test_accumulation_on_tutorial_with_ema(self):
        # Train with accum and without accum with same seeds
        # train loss must be equals
        accum = 10
        trainer_params = ScenarioTest.default_trainer_params()
        trainer_params.gen.setup.train.batch_size = accum
        trainer_params.gen.setup.train.limit = 1
        trainer_params.gen.setup.val.limit = 1
        trainer_params.epochs = 5
        trainer_params.learning_rate.lr = 0.01
        trainer_params.samples_per_epoch = accum
        trainer_params.skip_model_load_test = True
        trainer_params.train_accum_steps = 1
        trainer = ScenarioTest.create_trainer(trainer_params)
        first_train_logs = trainer.train()

        clear_session()
        trainer_params.gen.setup.train.batch_size = 1
        trainer_params.train_accum_steps = accum
        trainer_params.current_epoch = 0
        trainer_params.ema_decay = 0.9

        trainer = ScenarioTest.create_trainer(trainer_params)
        after_logs = trainer.train()

        # loss and acc on train must be equal, but lower on val
        self.assertAlmostEqual(first_train_logs["keras_loss"], after_logs["keras_loss"], places=2, msg="loss_loss")
        self.assertAlmostEqual(first_train_logs["acc"], after_logs["acc"], places=2, msg="acc")
        self.assertLess(first_train_logs["val_loss"], after_logs["val_loss"], "val_loss")
        self.assertGreater(first_train_logs["val_acc"], after_logs["val_acc"], "val_acc")

        clear_session()

    def test_accumulation_on_tutorial(self):
        # Train with accum and without accum with same seeds
        # train loss must be equals
        accum = 3
        trainer_params = ScenarioTest.default_trainer_params()
        trainer_params.gen.setup.train.batch_size = accum
        trainer_params.epochs = 3
        trainer_params.samples_per_epoch = accum
        trainer_params.train_accum_steps = 1
        trainer = ScenarioTest.create_trainer(trainer_params)
        first_train_logs = trainer.train()

        clear_session()
        trainer_params.gen.setup.train.batch_size = 1
        trainer_params.train_accum_steps = accum
        trainer_params.current_epoch = 0

        trainer = ScenarioTest.create_trainer(trainer_params)
        after_train_logs = trainer.train()

        for k, v in after_train_logs.items():
            self.assertAlmostEqual(
                v, first_train_logs[k], places=6, msg=f"{k}. Before {first_train_logs}, after {after_train_logs}"
            )
        clear_session()
