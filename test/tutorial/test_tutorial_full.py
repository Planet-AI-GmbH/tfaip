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

from examples.tutorial.full.data.data import TutorialData
from examples.tutorial.full.graphs.cnn import ConvGraphParams
from examples.tutorial.full.scenario import TutorialScenario
from test.util.training import resume_training, single_train_iter, lav_test_case, warmstart_training_test_case
from tfaip.data.databaseparams import DataPipelineParams


class TutorialScenarioTest(TutorialScenario):
    @classmethod
    def default_trainer_params(cls):
        p = super().default_trainer_params()
        p.gen.setup.train = DataPipelineParams(batch_size=1)
        p.gen.setup.val = DataPipelineParams(limit=5, batch_size=1)
        p.scenario.data.pre_proc.run_parallel = False
        return p


class TutorialWithConvScenarioTest(TutorialScenarioTest):
    @classmethod
    def default_trainer_params(cls):
        p = super().default_trainer_params()
        p.scenario.model.graph = ConvGraphParams()
        return p


class TestTutorialData(unittest.TestCase):
    def setUp(self) -> None:
        clear_session()

    def tearDown(self) -> None:
        clear_session()

    def test_data_loading(self):
        trainer_params = TutorialScenarioTest.default_trainer_params()
        data = TutorialData(trainer_params.scenario.data)
        with trainer_params.gen.train_data(data) as rd:
            train_data = next(rd.input_dataset().as_numpy_iterator())
        with trainer_params.gen.val_data(data) as rd:
            val_data = next(rd.input_dataset().as_numpy_iterator())

        def check(data):
            self.assertEqual(len(data), 3, "Expected (input, output, meta) tuple")
            self.assertEqual(len(data[0]), 1, "Expected one inputs")
            self.assertEqual(len(data[1]), 1, "Expected one outputs")
            self.assertEqual(len(data[2]), 1, "Expected one meta")
            self.assertTrue("img" in data[0])
            self.assertTrue("gt" in data[1])
            self.assertTrue("meta" in data[2])
            self.assertTupleEqual(data[0]["img"].shape, (1, 28, 28))
            self.assertTupleEqual(data[1]["gt"].shape, (1, 1))

        check(train_data)
        check(val_data)


class TestTutorialTrain(unittest.TestCase):
    scenario = TutorialWithConvScenarioTest

    def tearDown(self) -> None:
        clear_session()

    def test_single_train_iter(self):
        single_train_iter(self, self.scenario, debug=False)

    def test_resume_training(self):
        resume_training(self, self.scenario)

    def test_lav(self):
        lav_test_case(self, self.scenario, debug=False)

    def test_warmstart(self):
        warmstart_training_test_case(self, self.scenario)


if __name__ == "__main__":
    unittest.main()
