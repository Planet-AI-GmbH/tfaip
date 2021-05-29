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
import os
import tempfile
import unittest

from test.util.workdir import call_in_root

work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tutorial", "workdir"))


class TestLAVScript(unittest.TestCase):
    def test_lav_tutorial(self):
        with tempfile.TemporaryDirectory() as d:
            call_in_root(
                [
                    "tfaip-train",
                    "examples.tutorial.full",
                    "--trainer.samples_per_epoch",
                    "10",
                    "--trainer.epochs",
                    "1",
                    "--trainer.output_dir",
                    d,
                    "--train.batch_size",
                    "2",
                    "--val.limit",
                    "10",
                ]
            )
            call_in_root(
                [
                    "tfaip-lav",
                    "--export_dir",
                    os.path.join(d, "best"),
                    "--pipeline.limit",
                    "10",
                ]
            )
            call_in_root(
                [
                    "tfaip-lav",
                    "--export_dir",
                    os.path.join(d, "best"),
                    "--pipeline.limit",
                    "10",
                    "--data",
                    "examples.tutorial.full.data.prediction_data_generation:TutorialPredictionGeneratorParams",
                    "--data.files",
                    os.path.join(work_dir, "data", "*.png"),
                ]
            )
            call_in_root(
                [
                    "tfaip-lav",
                    "--export_dir",
                    os.path.join(d, "best"),
                    "--pipeline.limit",
                    "10",
                    "--data",
                    "examples.tutorial.full.data.prediction_data_generation:TutorialPredictionGeneratorParams",
                    "--data.files",
                    os.path.join(work_dir, "data", "*.png"),
                    "--run_eagerly",
                    "--dump",
                    os.path.join(d, "dump.pkl"),
                ]
            )

    def test_multi_lav_tutorial(self):
        with tempfile.TemporaryDirectory() as d:
            call_in_root(
                [
                    "tfaip-train",
                    "examples.tutorial.full",
                    "--trainer.samples_per_epoch",
                    "10",
                    "--trainer.epochs",
                    "1",
                    "--trainer.output_dir",
                    d,
                    "--train.batch_size",
                    "2",
                    "--val.limit",
                    "10",
                ]
            )
            call_in_root(
                [
                    "tfaip-multi-lav",
                    "--export_dirs",
                    os.path.join(d, "best"),
                    os.path.join(d, "best"),
                    "--data.files",
                    os.path.join(work_dir, "data", "*.png"),
                    "--pipeline.limit",
                    "10",
                ]
            )
