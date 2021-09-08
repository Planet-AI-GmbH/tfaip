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
"""Script to benchmark the throughput of the input pipeline of a scenario.

This script is similar to `tfaip-train` since the arguments are identical. Therefore, to execute the benchmark
just replace `tfaip-train` with `tfaip-benchmark-scenario-input-pipeline`.

"""
import logging
import sys
from argparse import Action
from contextlib import ExitStack
from typing import Type, TYPE_CHECKING

import numpy as np

from tfaip import TrainerParams
from tfaip.util.logging import WriteToLogFile
from tfaip.util.profiling import MeasureTime
from tfaip.util.tfaipargparse import TFAIPArgumentParser

if TYPE_CHECKING:
    from tfaip.imports import ScenarioBase

logger = logging.getLogger(__name__)


def run():
    main(*parse_args())


def main(scenario: Type["ScenarioBase"], trainer_params: TrainerParams, args):
    import tensorflow as tf

    with ExitStack() as stack:
        if trainer_params.output_dir:
            stack.enter_context(WriteToLogFile(trainer_params.output_dir, append=False))

        logger.info("trainer_params=" + trainer_params.to_json(indent=2))

        # create the trainer and run the benchmark
        trainer = scenario.create_trainer(trainer_params)
        trainer.setup_data()

        data = trainer.data
        training_pipeline = data.get_or_create_pipeline(trainer_params.gen.setup.train, trainer_params.gen.train_gen())

        print_interval = 0  # time until next print
        delta_print_interval = 0.5  # every half second
        with training_pipeline as rd:
            it = iter(enumerate(rd.input_dataset(auto_repeat=True).as_numpy_iterator()))
            next(it)  # take one example to initialize pipeline
            with MeasureTime() as total_time:
                tot_time_batch = 0
                tot_samples = 0
                tot_batches = 0
                last_10_times = []
                last_10_batch_sizes = []
                while args.time_limit < 0 or total_time.duration_till_now() < args.time_limit:
                    with MeasureTime() as batch_time:
                        i, sample = next(it)
                    batch_size = tf.nest.flatten(sample)[0].shape[0]
                    tot_time_batch += batch_time.duration
                    tot_samples += batch_size
                    tot_batches += 1
                    last_10_batch_sizes.append(batch_size)
                    last_10_times.append(batch_time.duration)
                    if len(last_10_batch_sizes) > 10:
                        del last_10_times[0]
                        del last_10_batch_sizes[0]

                    last_10_times_mean = sum(last_10_times)
                    print_interval -= batch_time.duration
                    if print_interval <= 0:
                        print_interval = delta_print_interval
                        sys.stdout.write(
                            f"Total time {total_time.duration_till_now() / 60:.3f}min - "
                            f"{tot_samples / tot_time_batch:.3f} samples / second (total)- "
                            f"{tot_batches / tot_time_batch:.3f} batches / second (total) - "
                            f"{sum(last_10_batch_sizes) / last_10_times_mean:.3f} samples / second (last_10) - "
                            f"{10 / last_10_times_mean:.3f} batches / second (last_10) - "
                            f"{batch_time.duration * batch_size:.3f} s / sample (last)"
                            f"{batch_time.duration:.3f} s / batch (last) - "
                            "\r"
                        )
                        sys.stdout.flush()


class ScenarioSelectionAction(Action):
    def __call__(self, parser: TFAIPArgumentParser, namespace, values, option_string=None):
        from tfaip.scenario.scenariobase import import_scenario

        scenario = import_scenario(values)

        # Now pass the real args of the scenario
        default_trainer_params = scenario.default_trainer_params()
        parser.add_root_argument("trainer", default_trainer_params.__class__, default=default_trainer_params)
        setattr(namespace, self.dest, scenario)


def parse_args(args=None):
    parser = TFAIPArgumentParser()

    parser.add_argument(
        "scenario_selection",
        help="Select the scenario by providing the module path which must be in the PYTHONPATH. "
        "Since a module is expected, separate with dots '.' not slashes. "
        "The module must either comprise a 'scenario.py' file with one "
        "scenario, else provide the full path to the Scenario class by separating the class name "
        "with a ':'. E.g. 'tfaip.scenario.tutorial.min', or "
        "'tfaip.scenario.tutorial.min.scenario:TutorialScenario'",
        action=ScenarioSelectionAction,
    )
    parser.add_argument(
        "--time_limit",
        default=-1,
        type=int,
        help="The number of seconds to run the test, or -1 for infinite (default).",
    )

    args = parser.parse_args(args=args)
    return args.scenario_selection, args.trainer, args


if __name__ == "__main__":
    run()
