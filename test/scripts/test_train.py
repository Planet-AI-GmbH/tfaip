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

from test.util.workdir import call_in_root

this_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.join(this_dir, "..", "..")


class TestTrainingScript(unittest.TestCase):
    def test_train_tutorial(self):
        call_in_root(
            [
                "tfaip-train",
                "examples.tutorial.full",  # tfaip is in PYTHONPATH so this import works
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

    def test_train_tutorial_with_class_name(self):
        call_in_root(
            [
                "tfaip-train",
                "examples.tutorial.full.scenario:TutorialScenario",
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

    def test_train_tutorial_python_path(self):
        env = os.environ.copy()
        sep = ";" if platform.system() == "Windows" else ":"
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] += sep + os.path.join(root_dir, "examples")
        else:
            env["PYTHONPATH"] = os.path.join(root_dir, "examples")

        call_in_root(
            [
                "tfaip-train",
                "tutorial.full",  # scenarios is in python path PYTHONPATH so this import works
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
            ],
            env=env,
        )

    def test_resume_train_tutorial(self):
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

            # train one more epoch (training was not cancelled)
            trainer_params = json.load(open(os.path.join(d, "trainer_params.json"), "r"))
            trainer_params["epochs"] = 2
            json.dump(trainer_params, open(os.path.join(d, "trainer_params.json"), "w"))
            call_in_root(["tfaip-resume-training", d])

    def test_train_from_params_tutorial(self):
        with tempfile.TemporaryDirectory() as d:
            call_in_root(
                [
                    "tfaip-train",
                    "examples.tutorial.full",
                    "--trainer.samples_per_epoch",
                    "1",
                    "--trainer.epochs",
                    "1",
                    "--trainer.output_dir",
                    d,
                    "--train.batch_size",
                    "1",
                    "--val.limit",
                    "10",
                ]
            )

            call_in_root(["tfaip-train-from-params", os.path.join(d, "trainer_params.json")])
