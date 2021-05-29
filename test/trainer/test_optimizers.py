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

from tensorflow.python.keras.backend import clear_session

from test.tutorial.test_tutorial_full import TutorialScenarioTest
from tfaip.data.databaseparams import DataPipelineParams
from tfaip.trainer.optimizer.optimizers import (
    OptimizerParams,
    SGDOptimizer,
    AdamOptimizer,
    AdamaxOptimizer,
    RMSpropOptimizer,
    AdaBeliefOptimizer,
)


class ScenarioTest(TutorialScenarioTest):
    @classmethod
    def default_trainer_params(cls):
        p = super().default_trainer_params()
        p.gen.setup.train = DataPipelineParams(batch_size=1)
        p.gen.setup.val = DataPipelineParams(limit=1, batch_size=1)
        p.gen.__post_init__()
        p.gen.train_val.dataset = "fashion_mnist"
        p.skip_model_load_test = True
        p.random_seed = 1337
        p.force_eager = False
        p.epochs = 1
        p.samples_per_epoch = 1
        return p


class TestOptimizers(unittest.TestCase):
    def tearDown(self) -> None:
        clear_session()

    def run_for_optimizer(self, optimizer: OptimizerParams):
        trainer_params = ScenarioTest.default_trainer_params()
        trainer_params.optimizer = optimizer
        trainer = ScenarioTest.create_trainer(trainer_params)
        trainer.train()
        # No asserts, just test if it runs

    def test_sgd_optimizer(self):
        self.run_for_optimizer(SGDOptimizer())

    def test_sgd_wd_optimizer(self):
        self.run_for_optimizer(SGDOptimizer(weight_decay=0.0001))

    def test_adam_optimizer(self):
        self.run_for_optimizer(AdamOptimizer())

    def test_adam_wd_optimizer(self):
        self.run_for_optimizer(AdamOptimizer(weight_decay=0.0001))

    def test_adamax_optimizer(self):
        self.run_for_optimizer(AdamaxOptimizer())

    def test_rmsprop_optimizer(self):
        self.run_for_optimizer(RMSpropOptimizer())

    def test_adabelief_optimizer(self):
        self.run_for_optimizer(AdaBeliefOptimizer())
