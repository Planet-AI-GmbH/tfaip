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
import json
import os
import sys
import tempfile
import time
import unittest
from typing import Type

import numpy as np
from tensorflow.keras.backend import clear_session

from test.util.store_logs_callback import StoreLogsCallback
from tfaip import WarmStartParams
from tfaip.scenario.scenariobase import ScenarioBase
from tfaip.trainer.trainer import Trainer
from tfaip.util.random import set_global_random_seed

debug_test = sys.flags.debug


def warmstart_training_test_case(test: unittest.TestCase, scenario: Type[ScenarioBase],
                                 debug=debug_test,
                                 delta=1E-5):
    # First train a normal iteration and store the results of metrics and losses with a fixed seed
    # Then reload the model as warmstart, train an epoch but with a learning rate of 0
    # The resulting metrics/loss must be identical
    with tempfile.TemporaryDirectory() as tmp_dir:
        store_logs_callback = StoreLogsCallback()
        trainer_params = scenario.default_trainer_params()
        trainer_params.gen.setup.val.limit = 1
        trainer_params.gen.setup.train.limit = 1
        trainer_params.output_dir = tmp_dir
        trainer_params.epochs = 1
        trainer_params.samples_per_epoch = trainer_params.gen.setup.train.batch_size
        trainer_params.skip_model_load_test = True  # not required in this test
        trainer_params.write_checkpoints = False
        trainer_params.export_best = True
        trainer_params.export_final = False
        trainer_params.random_seed = 1338  # Obtain same inputs from the input pipeline
        trainer_params.force_eager = debug
        trainer = scenario.create_trainer(trainer_params)
        trainer.train([store_logs_callback])

        initial_logs = store_logs_callback.logs
        # test loading from best
        clear_session()
        trainer_params.current_epoch = 0  # Restart training
        trainer_params.output_dir = None
        trainer_params.export_best = False
        trainer_params.warmstart = WarmStartParams(
            model=os.path.join(tmp_dir, 'best', 'serve')
        )
        trainer_params.learning_rate.lr = 0.0

        trainer = scenario.create_trainer(trainer_params)
        trainer.train([store_logs_callback])

        logs_after_warmstart = store_logs_callback.logs

        for k, v in logs_after_warmstart.items():
            if k.startswith('val'):
                # only test val variables, because training loss is once before and once after weight update
                test.assertAlmostEqual(v, initial_logs[k], delta=delta)


def single_train_iter(test: unittest.TestCase, scenario: Type[ScenarioBase], debug=debug_test):
    with tempfile.TemporaryDirectory() as tmp_dir:  # To write best model
        trainer_params = scenario.default_trainer_params()
        trainer_params.output_dir = tmp_dir
        trainer_params.scenario.debug_graph_construction = debug
        trainer_params.scenario.debug_graph_n_examples = 1
        trainer_params.epochs = 2
        trainer_params.samples_per_epoch = trainer_params.gen.setup.train.batch_size
        trainer_params.export_best = False
        trainer_params.export_final = True  # required, else due to magic circumstances, some tests will hang up...
        trainer_params.force_eager = debug
        trainer_params.random_seed = 1324
        trainer_params.lav_every_n = 1
        trainer_params.skip_model_load_test = debug
        trainer_params.val_every_n = 1

        trainer = scenario.create_trainer(trainer_params)
        trainer.train()


def lav_test_case(test: unittest.TestCase, scenario: Type[ScenarioBase], debug=False,
                  delta=1E-5,
                  batch_size_test=True,
                  ignore_binary_metric=False,
                  ignore_array_metric=False
                  ):
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer_params = scenario.default_trainer_params()
        trainer_params.output_dir = tmp_dir
        trainer_params.epochs = 1
        trainer_params.samples_per_epoch = 3
        trainer_params.skip_model_load_test = True  # not required in this test
        trainer_params.force_eager = debug
        trainer_params.export_best = True
        trainer_params.export_final = True
        trainer_params.write_checkpoints = False
        trainer_params.random_seed = 324
        trainer_params.scenario.data.pre_proc.run_parallel = False  # Deterministic results!
        batch_and_limit = 5
        trainer_params.scenario.debug_graph_construction = debug
        trainer = scenario.create_trainer(trainer_params)
        trainer.train()

        json_path = os.path.join(tmp_dir, 'trainer_params.json')
        with open(json_path) as f:
            trainer_params_dict = json.load(f)
        trainer_params_dict['epochs'] = 2

        lav_params = scenario.lav_cls().params_cls()()
        lav_params.model_path = os.path.join(trainer_params.output_dir, 'export')
        lav_params.pipeline = trainer_params.gen.setup.val
        clear_session()
        scenario_params = scenario.params_from_path(lav_params.model_path)
        lav = scenario.create_lav(lav_params, scenario_params)
        lav.run([trainer_params.gen.val_gen()])
        set_global_random_seed(trainer_params.random_seed)
        lav_params.model_path = os.path.join(trainer_params.output_dir, 'best')
        clear_session()
        scenario_params = scenario.params_from_path(lav_params.model_path)
        lav_params.pipeline.batch_size = 1
        lav_params.pipeline.limit = batch_and_limit
        lav = scenario.create_lav(lav_params, scenario_params)
        bs1_results = next(lav.run([trainer_params.gen.val_gen()], run_eagerly=debug))
        if batch_size_test:
            set_global_random_seed(trainer_params.random_seed)
            lav_params.model_path = os.path.join(trainer_params.output_dir, 'best')
            clear_session()
            scenario_params = scenario.params_from_path(lav_params.model_path)
            lav_params.pipeline.batch_size = batch_and_limit
            lav_params.pipeline.limit = batch_and_limit
            lav = scenario.create_lav(lav_params, scenario_params)
            bs5_results = next(lav.run([trainer_params.gen.val_gen()], run_eagerly=debug))
            time.sleep(0.5)
            for k in bs1_results.keys():
                if type(bs1_results[k]) == bytes:
                    if ignore_binary_metric:
                        continue
                    else:
                        test.assertEqual(bs1_results[k], bs5_results[k], msg=f"on key {k}")
                elif type(bs1_results[k]).__module__ == 'numpy':
                    if ignore_array_metric:
                        continue
                    else:
                        for x1, x5 in zip(np.reshape(bs1_results[k], [-1]), np.reshape(bs5_results[k], [-1])):
                            if str(bs1_results[k].dtype).startswith('int'):
                                test.assertEqual(x1, x5, msg=f"on key {k}")
                            else:
                                test.assertAlmostEqual(bs1_results[k], bs5_results[k], delta=delta, msg=f"on key {k}")
                else:
                    test.assertAlmostEqual(bs1_results[k], bs5_results[k], delta=delta, msg=f"on key {k}")


def resume_training(test: unittest.TestCase, scenario: Type[ScenarioBase], delta=1E-5, debug=debug_test):
    # simulate by setting epochs to 1, then loading the trainer_params and setting epochs to 2
    with tempfile.TemporaryDirectory() as tmp_dir:
        store_logs_callback = StoreLogsCallback()
        trainer_params = scenario.default_trainer_params()
        trainer_params.output_dir = tmp_dir
        trainer_params.epochs = 1
        trainer_params.samples_per_epoch = trainer_params.gen.setup.train.batch_size
        trainer_params.skip_model_load_test = True  # not required in this test
        trainer_params.force_eager = debug
        trainer_params.export_final = True  # due to magic circumstances, some tests will hang up if set to False...
        trainer_params.export_best = False
        trainer_params.random_seed = 1338 if trainer_params.random_seed is None else trainer_params.random_seed
        trainer = scenario.create_trainer(trainer_params)
        trainer.train([store_logs_callback])
        initial_logs = store_logs_callback.logs
        clear_session()

        json_path = os.path.join(tmp_dir, 'trainer_params.json')
        with open(json_path) as f:
            trainer_params_dict = json.load(f)

        # train another epoch
        # set learning rate to 0, thus. evaluation result must not change
        trainer_params_dict['epochs'] = 2
        trainer_params_dict['learning_rate']['lr'] = 0.0

        with open(json_path, 'w') as f:
            json.dump(trainer_params_dict, f)

        trainer = Trainer.restore_trainer(tmp_dir)
        trainer.train([store_logs_callback])
        logs_after_resume = store_logs_callback.logs

        for k, v in logs_after_resume.items():
            if k.startswith('val'):
                # only test val variables, because training loss is once before and once after weight update
                test.assertAlmostEqual(v, initial_logs[k], delta=delta)
