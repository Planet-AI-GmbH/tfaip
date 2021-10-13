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
import pickle
import tarfile
from argparse import Action

from tfaip import ScenarioBaseParams
from tfaip.scenario.scenariobase import ScenarioBase
from tfaip.util.multiprocessing.parallelmap import tqdm_wrapper
from tfaip.util.tfaipargparse import TFAIPArgumentParser

logger = logging.getLogger(__name__)


def run():
    logging.basicConfig(level=logging.INFO)
    main(*parse_args())


def main(prediction: str, scenario: ScenarioBase, scenario_params: ScenarioBaseParams):
    with scenario.create_evaluator(scenario_params.evaluator, scenario.static_evaluator_kwargs(scenario_params)) as e:
        with tarfile.open(prediction, "r:gz") as tar:
            names = tar.getnames()
            for file in tqdm_wrapper(tar.getnames(), total=len(names), desc="Evaluating", progress_bar=True):
                sample = pickle.load(tar.extractfile(file))
                e.update_state(sample)


class ScenarioSelectionAction(Action):
    def __call__(self, parser: TFAIPArgumentParser, namespace, values, option_string=None):
        scenario, scenario_params = ScenarioBase.from_path(values)
        evaluator_params = scenario_params.evaluator

        parser = TFAIPArgumentParser()
        parser.add_root_argument("evaluator", evaluator_params.__class__, evaluator_params)
        setattr(namespace, self.dest, scenario)
        setattr(namespace, self.dest + "_params", scenario_params)


def parse_args(args=None):
    parser = TFAIPArgumentParser()
    parser.add_argument("--prediction", type=str, required=True, help="Path to the prediction dump")
    parser.add_argument("--scenario", type=str, required=True, action=ScenarioSelectionAction)

    args = parser.parse_args(args=args)
    return args.prediction, args.scenario, args.scenario_params


if __name__ == "__main__":
    run()
