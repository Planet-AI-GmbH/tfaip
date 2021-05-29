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

from tfaip.imports import Trainer
from tfaip.util.logging import WriteToLogFile
from tfaip.util.tfaipargparse import TFAIPArgumentParser

logger = logging.getLogger(__name__)


class ScenarioSelectionAction(Action):
    def __call__(self, parser: TFAIPArgumentParser, namespace, values, option_string=None):
        output_dir = values
        trainer_params, scenario = Trainer.parse_trainer_params(output_dir)

        # parse additional args
        parser.add_root_argument("trainer", trainer_params.__class__, default=trainer_params)
        setattr(namespace, self.dest, values)
        setattr(namespace, "scenario_cls", scenario)


def main(args=None):
    parser = TFAIPArgumentParser()
    parser.add_argument(
        "output_dir", type=str, help="path to the checkpoint dir to resume from", action=ScenarioSelectionAction
    )
    args = parser.parse_args(args=args)

    with WriteToLogFile(args.trainer.output_dir, append=True):
        logger.info("=================================================================")
        logger.info(f"RESUMING TRAINING from {args.trainer.output_dir}")
        logger.info("=================================================================")

        # create the trainer
        trainer = args.scenario_cls.create_trainer(args.trainer, restore=True)
        trainer.train()


if __name__ == "__main__":
    main()
