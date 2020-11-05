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
from argparse import ArgumentParser
from tfaip.base.trainer import Trainer
import logging

from tfaip.util.logging import setup_log

logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser()

    parser.add_argument('checkpoint_dir', type=str, help='path to the checkpoint dir to resume from')

    args = parser.parse_args()
    setup_log(args.checkpoint_dir, append=True)

    logger.info("=================================================================")
    logger.info(f"RESUMING TRAINING from {args.checkpoint_dir}")
    logger.info("=================================================================")

    trainer = Trainer.restore_trainer(args.checkpoint_dir)
    trainer.train()


if __name__ == '__main__':
    main()
