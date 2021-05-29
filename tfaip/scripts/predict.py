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
import io
import logging
import os
import pickle
import tarfile
import time
from argparse import Action

from tfaip.imports import ScenarioBase
from tfaip.util.tfaipargparse import TFAIPArgumentParser

logger = logging.getLogger(__name__)


def run():
    logging.basicConfig(level=logging.INFO)
    main(*parse_args())


def main(args, scenario_cls: ScenarioBase, scenario_params):
    predict_params = args.predict
    logger.info("data_params=" + scenario_params.data.to_json(indent=2))
    logger.info("predict_params=" + predict_params.to_json(indent=2))

    # create the predictor and data and run it
    if len(args.export_dir) == 1:
        predictor = scenario_cls.create_predictor(args.export_dir[0], predict_params)
    elif len(args.export_dir) > 1:
        predictor = scenario_cls.create_multi_predictor(args.export_dir, predict_params)
    else:
        raise ValueError("At least one export dir is required.")

    prediction_gen = enumerate(predictor.predict(args.data))
    if args.dump_prediction:
        path = args.dump_prediction + ".tar.gz"
        if os.path.isfile(path):
            os.remove(path)
        with tarfile.open(path, mode="x:gz") as tar:
            for i, r in prediction_gen:
                b = io.BytesIO()
                pickle.dump(r, b)
                b.seek(0)
                tarinfo = tarfile.TarInfo(name=f"{i}.pred")
                tarinfo.mtime = time.time()
                tarinfo.size = len(b.getvalue())
                tar.addfile(tarinfo, fileobj=b)
    else:
        for i, r in prediction_gen:
            pass

    predictor.benchmark_results.pretty_print()


class ScenarioSelectionAction(Action):
    def __call__(self, parser: TFAIPArgumentParser, namespace, values, option_string=None):
        scenario, scenario_params = ScenarioBase.from_path(values[0])
        predict_params = scenario.predictor_cls().params_cls()()
        parser.add_root_argument("data", scenario.predict_generator_params_cls())
        parser.add_root_argument("predict", predict_params.__class__, default=predict_params)

        setattr(namespace, self.dest, values)
        setattr(namespace, "scenario", scenario)
        setattr(namespace, "scenario_params", scenario_params)


def parse_args(args=None):
    parser = TFAIPArgumentParser()
    parser.add_argument("--export_dir", required=True, nargs="+", action=ScenarioSelectionAction)
    parser.add_argument("--dump_prediction", type=str, help="Dumps the prediction results as tar.gz")

    args = parser.parse_args(args=args)
    return args, args.scenario, args.scenario_params


if __name__ == "__main__":
    run()
