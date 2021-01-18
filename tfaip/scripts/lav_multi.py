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
import logging

from tfaip.base.lav.callbacks.dump_results import DumpResultsCallback
from tfaip.util.argumentparser.parser import add_args_group, TFAIPArgumentParser


logger = logging.getLogger(__name__)


def run():
    logging.basicConfig(level=logging.INFO)
    main(*parse_args())


def main(args, scenario_cls, scenario_params, predictor_params):
    callbacks = []
    if args.dump:
        callbacks.append(DumpResultsCallback(args.dump))

    lav_params = args.lav_params
    logger.info("data_params=" + scenario_params.data_params.to_json(indent=2))
    logger.info("lav_params=" + lav_params.to_json(indent=2))

    # create the lav and run it
    lav = scenario_cls.create_multi_lav(lav_params, scenario_params, predictor_params)
    for i, r in enumerate(lav.run(run_eagerly=args.run_eagerly, callbacks=callbacks)):
        print(json.dumps(r, indent=2))
        lav.benchmark_results.pretty_print()


def parse_args(args=None):
    from tfaip.base.scenario.scenariobase import ScenarioBase

    parser = TFAIPArgumentParser()
    parser.add_argument('--export_dirs', required=True, nargs='+')
    parser.add_argument('--run_eagerly', action='store_true', help="Run the graph in eager mode. This is helpful for debugging. Note that all custom layers must be added to ModelBase!")
    parser.add_argument('--dump', type=str, help='Dump the predictions and results to the given filepath')

    args, unknown_args = parser.parse_known_args(args)
    scenario, scenario_params = ScenarioBase.from_path(args.export_dirs[0])  # scenario based on first model
    pipeline_params = scenario_params.data_params.val
    lav_params = scenario.lav_cls().get_params_cls()()
    lav_params.model_path = args.export_dirs
    predictor_params = scenario.multi_predictor_cls().get_params_cls()()

    parser = TFAIPArgumentParser()
    add_args_group(parser, group='lav_params', default=lav_params, params_cls=scenario.lav_cls().get_params_cls())
    add_args_group(parser, group='predictor_params', default=predictor_params, params_cls=scenario.multi_predictor_cls().get_params_cls(),
                   exclude_field_names={'device_params', 'silent', 'progress_bar', 'run_eagerly', 'include_targets'})
    add_args_group(parser, group='data', default=pipeline_params, params_cls=pipeline_params.__class__)

    return parser.parse_args(unknown_args, namespace=args), scenario, scenario_params, predictor_params


if __name__ == '__main__':
    run()
