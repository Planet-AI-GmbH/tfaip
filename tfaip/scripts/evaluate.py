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
import io
import logging
import os
import tarfile
import pickle
import time

from tfaip.scenario import scenarios, get_scenario_by_name
from tfaip.util.argument_parser import TFAIPArgumentParser
from tfaip.util.multiprocessing.parallelmap import tqdm_wrapper

logger = logging.getLogger(__name__)


def run():
    logging.basicConfig(level=logging.INFO)
    main(parse_args())


def main(args):
    scenario = get_scenario_by_name(args.scenario).scenario
    with scenario.create_evaluator() as e:
        with tarfile.open(args.prediction, 'r:gz') as tar:
            names = tar.getnames()
            for file in tqdm_wrapper(tar.getnames(), total=len(names), desc='Evaluating', progress_bar=True):
                sample = pickle.load(tar.extractfile(file))
                e.handle(sample)


def parse_args():
    parser = TFAIPArgumentParser()
    parser.add_argument('--prediction', type=str, required=True, help="Path to the prediction dump")
    parser.add_argument('--scenario', type=str, required=True)

    return parser.parse_args()


if __name__ == '__main__':
    run()
