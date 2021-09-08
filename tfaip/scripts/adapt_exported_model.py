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
import logging
import os
from argparse import Action

from tfaip.util.tfaipargparse import TFAIPArgumentParser

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class ScenarioSelectionAction(Action):
    def __call__(self, parser: TFAIPArgumentParser, namespace, values, option_string=None):
        from tfaip.imports import ScenarioBase

        export_dir = values
        scenario, scenario_params = ScenarioBase.from_path(export_dir)

        parser.add_root_argument("scenario", scenario_params.__class__, default=scenario_params)
        setattr(namespace, self.dest, values)
        setattr(namespace, "scenario_cls", scenario)


def run():
    parser = TFAIPArgumentParser(
        description="Program to update older graphs to newer version."
        "Recreate the Graph of a model based on the current code and optional"
        "changed parameters, then loads the weights, and reexports the model "
        "with the adapted settings and code."
        "Note: the weights must be unchanged!"
        "This is for example useful to adapt a parameter in the stored model, e.g. "
        "the beam with or a weighting factor: --model.inference_decoder.beam_width=5"
    )

    parser.add_argument("export_dir", action=ScenarioSelectionAction, help="path to the checkpoint dir to resume from")
    parser.add_argument("--output_dir", help="path where to write the model to")
    parser.add_argument("--overwrite", help="overwrite the existing model", action="store_true")
    parser.add_argument(
        "--no_check_loaded",
        help="do not check if the loaded weights match with the existing ones."
        "This is particularly useful if you change the size of the model, "
        "e.g. by adding a pretrained language model",
        action="store_true",
    )

    args = parser.parse_args()

    assert args.output_dir or args.overwrite, "either or required"
    if args.overwrite:
        args.output_dir = args.export_dir

    scenario_cls = args.scenario_cls

    trainer_params = scenario_cls.default_trainer_params()
    trainer_params.scenario = args.scenario
    trainer_params.scenario.print_eval_limit = 0

    scenario = trainer_params.scenario.create()
    # setup training and prediction graph with dummy settings
    scenario.setup_training("adam", skip_model_load_test=True)

    def store(path, to_path):
        assert path is not None
        assert to_path is not None
        print(f"Converting from {path} to {to_path}")
        r = scenario.keras_predict_model.load_weights(os.path.join(path, "serve", "variables", "variables"))
        if not args.no_check_loaded:
            r.assert_existing_objects_matched()
            print("Variables successfully loaded. All existing objects matched.")
        else:
            print("Skipping verification to check if all variables are present in the provided checkpoint.")
        print(f"Attempting to store new model to {to_path}")
        scenario.export(to_path, export_resources=path != to_path)
        print(f"{to_path} successfully written.")

    store(args.export_dir, args.output_dir)


if __name__ == "__main__":
    run()
