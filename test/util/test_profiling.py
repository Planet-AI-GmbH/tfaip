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

from tfaip.util.profiling import profile, print_profiling, ProfileScope, enable_profiling


class Test(TestCase):
    def test_print_profiling(self):
        # enable_profiling(True)

        @profile
        def function_1(an_arr: np.ndarray) -> np.ndarray:
            return an_arr ** 5

        @profile
        def function_2(an_arr: np.ndarray) -> float:
            return function_1(function_1(an_arr))

        dim = 300
        inarr: np.ndarray = np.random.random([dim] * 3)
        inarr2: np.ndarray = np.random.random([dim] * 3)

        with ProfileScope(self.__module__ + "::run it twice"):
            res3 = function_2(function_2(inarr))
        #
        res1 = function_1(inarr2)
        res2 = function_2(inarr2)

        print(np.sum(res1))
        print(np.sum(res2))
        print(np.sum(res3))
        print_profiling()
