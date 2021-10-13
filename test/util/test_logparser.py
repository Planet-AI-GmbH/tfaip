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
from tfaip.util.logging import ParseLogFile
import os

this_dir = os.path.dirname(os.path.realpath(__file__))
test_util_dir = os.path.abspath(os.path.join(this_dir, "..", "..", "test", "util"))


class TestLogParser(unittest.TestCase):
    def test_get_metrics_dict(self):
        log_dir = test_util_dir
        log_name = "example_train.log"
        parser = ParseLogFile(log_dir, log_name=log_name)
        metrics = parser.get_metrics()

        # check length of output
        self.assertEqual(len(metrics), 8)
        dict_ref = {
            "loss": "0.5049",
            "loss/mean_epoch": "0.5049",
            "mae": "0.4960",
            "mse": "0.5049",
            "val_loss": "0.4632",
            "val_loss/mean_epoch": "0.4632",
            "val_mae": "0.4775",
            "val_mse": "0.4632",
        }
        self.assertDictEqual(metrics, dict_ref)

    def test_file_error(self):
        log_dir = "."
        parser = ParseLogFile(log_dir, log_name="not_existent.log")
        self.assertDictEqual(parser.get_metrics(), dict())


if __name__ == "__main__":
    unittest.main()
