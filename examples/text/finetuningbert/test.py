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

from tensorflow.python.keras.backend import clear_session

from examples.text.finetuningbert.scenario import FTBertScenario
from tfaip.util.testing.training import single_train_iter
from tfaip.util.testing.workdir import call_in_root

this_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.abspath(os.path.join(this_dir, "..", ".."))


class FTBertTestScenario(FTBertScenario):
    @classmethod
    def default_trainer_params(cls):
        params = super().default_trainer_params()
        return params


class TestFineTuningBert(unittest.TestCase):
    scenario = FTBertTestScenario

    def tearDown(self) -> None:
        clear_session()

    def test_single_train_iter(self):
        single_train_iter(self, self.scenario, debug=False)

    def test_cmd_line_of_readme(self):
        # Keep this in sync with README.md
        with tempfile.TemporaryDirectory() as ftbert_model:
            for cmd in [
                f"tfaip-train examples.text.finetuningbert --trainer.output_dir {ftbert_model}",
            ]:
                additional_args = []
                if "tfaip-train" in cmd:
                    additional_args = ["--trainer.epochs", "1", "--trainer.samples_per_epoch", "32"]

                call_in_root(cmd.split(" ") + additional_args)
