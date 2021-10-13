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

import numpy as np

from examples.howtos.differenttrainandvalgraph.graph import MyModelParams


class TestDifferentTrainAndValGraph(TestCase):
    def test_graph(self):
        graph = MyModelParams().create_graph()
        data = np.random.random((10, 10))

        train_out = graph.train(inputs=data, targets=None)
        predict_out = graph.predict(inputs=data)

        np.testing.assert_allclose(train_out / 2, predict_out)
