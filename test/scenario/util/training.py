import json
import os
import tempfile
import time
import unittest
from typing import Type

from tensorflow.python.keras.backend import clear_session

from test.scenario.util.store_logs_callback import StoreLogsCallback
from tfaip.base.scenario import ScenarioBaseParams, ScenarioBase
from tfaip.base.trainer import TrainerParams, Trainer
from tfaip.base.trainer.warmstart.warmstart_params import WarmstartParams
from tfaip.util.random import set_global_random_seed


def warmstart_training_test_case(test: unittest.TestCase, scenario, scenario_params: ScenarioBaseParams, debug=True):
    # First train a normal iteration and store the results of metrics and losses with a fixed seed
    # Then reload the model as warmstart, train an epoch but with a learning rate of 0
    # The resulting metrics/loss must be identical
    with tempfile.TemporaryDirectory() as tmp_dir:
        store_logs_callback = StoreLogsCallback()
        scenario_params.data_params.val_limit = 1  # Force the same example
        scenario_params.data_params.train_limit = 1  # Force the same example
        trainer_params = TrainerParams(
            checkpoint_dir=tmp_dir,
            epochs=1,
            samples_per_epoch=scenario_params.data_params.train_batch_size,
            skip_model_load_test=True,  # not required in this test
            scenario_params=scenario_params,
            write_checkpoints=False,
            export_best=True,
            export_final=False,
            random_seed=1338,  # Obtain same inputs from the input pipeline
            force_eager=debug,
        )
        trainer = scenario.create_trainer(trainer_params)
        trainer.train([store_logs_callback])

        initial_logs = store_logs_callback.logs
        # test loading from best
        clear_session()
        trainer_params.current_epoch = 0  # Restart training
        trainer_params.checkpoint_dir = None
        trainer_params.export_best = False
        trainer_params.warmstart_params = WarmstartParams(
            model=os.path.join(tmp_dir, 'best', 'serve')
        )
        trainer_params.learning_rate_params.lr = 0.0

        trainer = scenario.create_trainer(trainer_params)
        trainer.train([store_logs_callback])

        logs_after_warmstart = store_logs_callback.logs

        for k, v in logs_after_warmstart.items():
            if k.startswith('val'):
                # only test val variables, because training loss is once before and once after weight update
                test.assertAlmostEqual(v, initial_logs[k])


def single_train_iter(test: unittest.TestCase, scenario, scenario_params: ScenarioBaseParams, debug=True):
    scenario_params.debug_graph_construction = debug
    scenario_params.debug_graph_n_examples = 1
    trainer_params = TrainerParams(
        epochs=1,
        samples_per_epoch=scenario_params.data_params.train_batch_size,
        scenario_params=scenario_params,
        write_checkpoints=False,
        force_eager=debug,
        random_seed=1324,
        lav_every_n=1,
    )
    trainer = scenario.create_trainer(trainer_params)
    trainer.train()


def lav_test_case(test: unittest.TestCase, scenario: Type[ScenarioBase], scenario_params, debug=True):
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer_params = TrainerParams(
            checkpoint_dir=tmp_dir,
            epochs=1,
            samples_per_epoch=3,
            skip_model_load_test=True,  # not required in this test
            scenario_params=scenario_params,
            force_eager=debug,
            export_best=True,
            export_final=True,
            write_checkpoints=False,
            random_seed=324,
        )
        trainer = scenario.create_trainer(trainer_params)
        trainer.train()

        json_path = os.path.join(tmp_dir, 'trainer_params.json')
        with open(json_path) as f:
            trainer_params_dict = json.load(f)
        trainer_params_dict['epochs'] = 2

        lav_params = scenario.lav_cls().get_params_cls()()
        lav_params.max_iter = 1

        lav_params.model_path_ = os.path.join(trainer_params.checkpoint_dir, 'export', 'serve')
        clear_session()
        lav = scenario.create_lav(lav_params, scenario_params)
        lav.run()
        set_global_random_seed(trainer_params.random_seed)
        lav_params.max_iter = 5
        lav_params.model_path_ = os.path.join(trainer_params.checkpoint_dir, 'best', 'serve')
        clear_session()
        scenario_params.data_params.val_batch_size = 1
        lav = scenario.create_lav(lav_params, scenario_params)
        bs1_results = next(lav.run())
        set_global_random_seed(trainer_params.random_seed)
        lav_params.max_iter = 1
        lav_params.model_path_ = os.path.join(trainer_params.checkpoint_dir, 'best', 'serve')
        clear_session()
        scenario_params.data_params.val_batch_size = 5
        lav = scenario.create_lav(lav_params, scenario_params)
        bs5_results = next(lav.run())
        time.sleep(0.5)
        for k in bs1_results.keys():
            test.assertAlmostEqual(bs1_results[k], bs5_results[k], msg=f"on key {k}")


def resume_training(test: unittest.TestCase, scenario, scenario_params):
    # simulate by setting epochs to 1, then loading the trainer_params and setting epochs to 2
    with tempfile.TemporaryDirectory() as tmp_dir:
        store_logs_callback = StoreLogsCallback()
        trainer_params = TrainerParams(
            checkpoint_dir=tmp_dir,
            epochs=1,
            samples_per_epoch=scenario_params.data_params.train_batch_size,
            skip_model_load_test=True,  # not required in this test
            export_final=False,
            export_best=False,
            scenario_params=scenario_params,
            random_seed=1338,
        )
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
        trainer_params_dict['learning_rate_params']['lr'] = 0.0

        with open(json_path, 'w') as f:
            json.dump(trainer_params_dict, f)

        trainer = Trainer.restore_trainer(tmp_dir)
        trainer.train([store_logs_callback])
        logs_after_resume = store_logs_callback.logs

        for k, v in logs_after_resume.items():
            if k.startswith('val'):
                # only test val variables, because training loss is once before and once after weight update
                test.assertAlmostEqual(v, initial_logs[k])
