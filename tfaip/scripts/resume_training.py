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
from tfaip.base.imports import Trainer
import logging

from tfaip.util.argumentparser import TFAIPArgumentParser, add_args_group
from tfaip.util.logging import setup_log

logger = logging.getLogger(__name__)


def main():
    parser = TFAIPArgumentParser()

    parser.add_argument('checkpoint_dir', type=str, help='path to the checkpoint dir to resume from')

    args, unknown_args = parser.parse_known_args()
    setup_log(args.checkpoint_dir, append=True)

    logger.info("=================================================================")
    logger.info(f"RESUMING TRAINING from {args.checkpoint_dir}")
    logger.info("=================================================================")

    trainer_params, scenario = Trainer.parse_trainer_params(args.checkpoint_dir)

    # parse additional args
    parser = TFAIPArgumentParser(ignore_required=True)
    add_args_group(parser, group='trainer_params', default=trainer_params, params_cls=trainer_params)
    parser.parse_args(unknown_args)

    # create the trainer
    trainer = scenario.create_trainer(trainer_params, restore=True)
    trainer.train()


if __name__ == '__main__':
    main()
