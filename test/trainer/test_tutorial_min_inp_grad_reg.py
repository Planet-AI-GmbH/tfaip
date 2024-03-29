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

from test.examples.tutorial.test_tutorial_full import TutorialScenarioTest
from tfaip.util.testing.training import single_train_iter, AdditionalTrainerArgs


class TestTutorialTrain(unittest.TestCase):
    def tearDown(self) -> None:
        clear_session()

    def test_single_train_iter(self):
        single_train_iter(
            self, TutorialScenarioTest, debug=False, args=AdditionalTrainerArgs(input_gradient_regularization=True)
        )


if __name__ == "__main__":
    unittest.main()
