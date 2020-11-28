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
import json

from tfaip.base.trainer import Trainer
import logging

from tfaip.util.argument_parser import TFAIPArgumentParser
from tfaip.util.logging import setup_log

logger = logging.getLogger(__name__)


def main():
    parser = TFAIPArgumentParser()
    parser.add_argument('params_file', type=str, help='path to the trainer_params.json')
    args = parser.parse_args()

    with open(args.params_file) as f:
        trainer_params = json.load(f)

    setup_log(trainer_params['checkpoint_dir'], False)

    trainer = Trainer.trainer_from_dict(trainer_params)
    trainer.train()


if __name__ == '__main__':
    main()
