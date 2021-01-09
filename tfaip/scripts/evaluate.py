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
import logging
import tarfile
import pickle

from tfaip.scenario import get_scenario_by_name
from tfaip.util.argumentparser.parser import TFAIPArgumentParser, add_args_group
from tfaip.util.multiprocessing.parallelmap import tqdm_wrapper

logger = logging.getLogger(__name__)


def run():
    logging.basicConfig(level=logging.INFO)
    main(*parse_args())


def main(prediction: str, scenario, evaluator_params):
    with scenario.create_evaluator(evaluator_params) as e:
        with tarfile.open(prediction, 'r:gz') as tar:
            names = tar.getnames()
            for file in tqdm_wrapper(tar.getnames(), total=len(names), desc='Evaluating', progress_bar=True):
                sample = pickle.load(tar.extractfile(file))
                e.update_state(sample)


def parse_args():
    parser = TFAIPArgumentParser()
    parser.add_argument('--prediction', type=str, required=True, help="Path to the prediction dump")
    parser.add_argument('--scenario', type=str, required=True)

    args, unknown_args = parser.parse_known_args()
    scenario = get_scenario_by_name(args.scenario).scenario
    evaluator_params = scenario.evaluator_cls().default_params()

    parser = TFAIPArgumentParser()
    add_args_group(parser, 'evaluation_params', evaluator_params, evaluator_params)
    parser.parse_args(unknown_args)

    return args.prediction, scenario, evaluator_params


if __name__ == '__main__':
    run()
