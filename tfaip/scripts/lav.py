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
import os
from argparse import ArgumentParser

from tfaip.base.lav.callbacks.dump_results import DumpResultsCallback
from tfaip.util.argument_parser import add_args_group

import logging


logger = logging.getLogger(__name__)


def run():
    logging.basicConfig(level=logging.INFO)
    main(*parse_args())


def main(args, scenario_meta, scenario_params):
    callbacks = []
    if args.dump:
        callbacks.append(DumpResultsCallback(args.dump))

    lav_params = args.lav_params
    logger.info("data_params=" + scenario_params.data_params.to_json(indent=2))
    logger.info("lav_params=" + lav_params.to_json(indent=2))

    # create the lav and run it
    lav = scenario_meta.create_lav(lav_params, scenario_params)
    for i, r in enumerate(lav.run(run_eagerly=args.run_eagerly, callbacks=callbacks)):
        lav.benchmark_results.pretty_print()


def parse_args(args=None):
    from tfaip.base.scenario import ScenarioBase
    parser = ArgumentParser()
    parser.add_argument('--export_dir', required=True)
    parser.add_argument('--run_eagerly', action='store_true', help="Run the graph in eager mode. This is helpful for debugging. Note that all custom layers must be added to ModelBase!")
    parser.add_argument('--dump', type=str, help='Dump the predictions and results to the given filepath')

    args, unknown_args = parser.parse_known_args(args)
    scenario, scenario_params = ScenarioBase.from_path(args.export_dir)

    lav_params = scenario.lav_cls().get_params_cls()()
    lav_params.model_path_ = args.export_dir

    parser = ArgumentParser()
    add_args_group(parser, group='data_params', default=scenario_params.data_params, params_cls=scenario.data_cls().get_params_cls())
    add_args_group(parser, group='lav_params', default=lav_params, params_cls=scenario.lav_cls().get_params_cls())

    return parser.parse_args(unknown_args, namespace=args), scenario, scenario_params


if __name__ == '__main__':
    run()
