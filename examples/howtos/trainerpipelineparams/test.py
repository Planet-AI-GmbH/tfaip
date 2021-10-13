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
from unittest import TestCase

from examples.howtos.trainerpipelineparams.scenario import MyScenario
from tfaip import PipelineMode


class TestTrainerPipelineParamsScenario(TestCase):
    def test_instance(self):
        # Create the training generator (default, 100 numbers with 80/20 ratio)
        train = list(MyScenario.default_trainer_params().gen.train_gen().create(PipelineMode.TRAINING).generate())
        self.assertListEqual(list(range(20, 100)), train)

        val = list(MyScenario.default_trainer_params().gen.val_gen().create(PipelineMode.EVALUATION).generate())
        self.assertListEqual(list(range(0, 20)), val)
