# Copyright 2020 The tfaip authors. All Rights Reserved.
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


class TestParallelData(unittest.TestCase):
    def test_run(self):
        from test.base.test_parallel_data.pipeline import Pipeline
        from tfaip.base.data.data import DataBase, DataBaseParams

        class TestData(DataBase):
            def _get_train_data(self):
                pipeline = Pipeline(self, 8, 1000)
                return pipeline.output_generator()

            def _get_val_data(self):
                pass

            def _input_layer_specs(self):
                pass

            def _target_layer_specs(self):
                pass

        params = DataBaseParams()
        data = TestData(params)
        with data:
            for i, d in enumerate(zip(data._get_train_data(), range(100))):
                pass

