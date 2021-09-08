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
import json
import os
import platform
import tempfile
import unittest

from tfaip.util.testing.workdir import call_in_root

this_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.join(this_dir, "..", "..")


class TestBenchmarkScenarioInputPipelineScript(unittest.TestCase):
    def test_tutorial(self):
        call_in_root(
            [
                "tfaip-benchmark-scenario-input-pipeline",
                "examples.tutorial.full",  # tfaip is in PYTHONPATH so this import works
                "--time_limit",
                "1",
                "--trainer.samples_per_epoch",
                "10",
                "--trainer.epochs",
                "2",
                "--train.batch_size",
                "2",
                "--val.limit",
                "10",
                "--model.graph",
                "MLPGraph",  # Check selection without 'Params'-suffix
            ]
        )
