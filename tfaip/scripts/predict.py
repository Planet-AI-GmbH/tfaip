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
from argparse import ArgumentParser

from tfaip.util.argument_parser import add_args_group
from tfaip.base.scenario import ScenarioBase


logger = logging.getLogger(__name__)


def run():
    logging.basicConfig(level=logging.INFO)
    main(*parse_args())


def main(args, scenario_cls: ScenarioBase, scenario_params):
    predict_params = args.predict_params
    logger.info("data_params=" + scenario_params.data_params.to_json(indent=2))
    logger.info("predict_params=" + args.predict_params.to_json(indent=2))

    # create the predictor and data and run it
    predictor = scenario_cls.create_predictor(predict_params, scenario_params)
    for i, r in enumerate(predictor.predict_lists(args.predict_lists)):
        pass

    predictor.benchmark_results.pretty_print()


def parse_args(args=None):
    parser = ArgumentParser()
    parser.add_argument('--export_dir', required=True)
    parser.add_argument('--predict_lists', required=True, nargs='+')

    args, unknown_args = parser.parse_known_args(args)
    scenario, scenario_params = ScenarioBase.from_path(args.export_dir)

    predict_params = scenario.predictor_cls().get_params_cls()()
    predict_params.model_path_ = args.export_dir

    parser = ArgumentParser()
    add_args_group(parser, group='data_params', default=scenario_params.data_params, params_cls=scenario.data_cls().get_params_cls())
    add_args_group(parser, group='predict_params', default=predict_params, params_cls=scenario.predictor_cls().get_params_cls())

    return parser.parse_args(unknown_args, namespace=args), scenario, scenario_params


if __name__ == '__main__':
    run()
