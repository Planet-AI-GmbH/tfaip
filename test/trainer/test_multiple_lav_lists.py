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
import tempfile
import unittest

from tensorflow.keras.backend import clear_session

from test.tutorial.test_tutorial_full import TutorialScenarioTest
from tfaip.data.databaseparams import DataPipelineParams


class ScenarioTest(TutorialScenarioTest):
    @classmethod
    def default_trainer_params(cls):
        p = super().default_trainer_params()
        p.gen.setup.train = DataPipelineParams(limit=50, batch_size=1)
        p.gen.setup.val = DataPipelineParams(limit=50, batch_size=1)
        p.gen.__post_init__()
        p.skip_model_load_test = True
        p.random_seed = 1337
        p.force_eager = False
        p.epochs = 1
        p.samples_per_epoch = 1
        p.lav_every_n = 1
        return p


class TestMultipleValLists(unittest.TestCase):
    def setUp(self) -> None:
        clear_session()

    def tearDown(self) -> None:
        clear_session()

    def test_lav_during_training(self):
        with tempfile.TemporaryDirectory() as d:
            # Train with ema and without ema with same seeds
            # train loss must be equals, but with ema the validation outcomes must be different
            trainer_params = ScenarioTest.default_trainer_params()
            trainer_params.output_dir = d
            trainer = ScenarioTest.create_trainer(trainer_params)
            train_logs = trainer.train()
            # Tutorial yields two LAV datasets (test and train)
            for i in range(2):
                self.assertAlmostEqual(train_logs[f"lav_l{i}_acc"], train_logs[f"lav_l{i}_eval_acc"])
