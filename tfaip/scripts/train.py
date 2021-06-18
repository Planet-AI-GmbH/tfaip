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
from argparse import Action
from contextlib import ExitStack
from typing import Type, TYPE_CHECKING

from tfaip import TrainerParams
from tfaip.util.logging import WriteToLogFile
from tfaip.util.tfaipargparse import TFAIPArgumentParser

if TYPE_CHECKING:
    from tfaip.imports import ScenarioBase

logger = logging.getLogger(__name__)


def run():
    main(*parse_args())


def main(scenario: Type["ScenarioBase"], trainer_params: TrainerParams):
    with ExitStack() as stack:
        if trainer_params.output_dir:
            stack.enter_context(WriteToLogFile(trainer_params.output_dir, append=False))

        logger.info("trainer_params=" + trainer_params.to_json(indent=2))

        # create the trainer and run it
        trainer = scenario.create_trainer(trainer_params)
        trainer.train()


class ScenarioSelectionAction(Action):
    def __call__(self, parser: TFAIPArgumentParser, namespace, values, option_string=None):
        from tfaip.scenario.scenariobase import import_scenario

        scenario = import_scenario(values)

        # Now pass the real args of the scenario
        default_trainer_params = scenario.default_trainer_params()
        parser.add_root_argument("trainer", default_trainer_params.__class__, default=default_trainer_params)
        setattr(namespace, self.dest, scenario)


def parse_args(args=None):
    parser = TFAIPArgumentParser()

    parser.add_argument(
        "scenario_selection",
        help="Select the scenario by providing the module path which must be in the PYTHONPATH. "
        "Since a module is expected, separate with dots '.' not slashes. "
        "The module must either comprise a 'scenario.py' file with one "
        "scenario, else provide the full path to the Scenario class by separating the class name "
        "with a ':'. E.g. 'tfaip.scenario.tutorial.min', or "
        "'tfaip.scenario.tutorial.min.scenario:TutorialScenario'",
        action=ScenarioSelectionAction,
    )

    args = parser.parse_args(args=args)
    return args.scenario_selection, args.trainer


if __name__ == "__main__":
    run()
