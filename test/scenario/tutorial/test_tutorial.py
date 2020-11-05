import unittest

from tensorflow.python.keras.backend import clear_session

from test.util.training import resume_training, single_train_iter, lav_test_case, warmstart_training_test_case
from tfaip.scenario.tutorial.data import DataParams, Data
from tfaip.scenario.tutorial.scenario import TutorialScenario


def get_default_data_params():
    return DataParams(
        train_batch_size=1,
        val_batch_size=1,
        val_limit=5,
    )


def get_default_scenario_params():
    default_params = TutorialScenario.default_params()
    default_params.data_params = get_default_data_params()
    return default_params


class TestTutorialData(unittest.TestCase):
    def setUp(self) -> None:
        clear_session()

    def test_data_loading(self):
        data = Data(get_default_data_params())
        with data:
            train_data = next(data.get_train_data().as_numpy_iterator())
            val_data = next(data.get_val_data().as_numpy_iterator())

            def check(data):
                self.assertEqual(len(data), 2, "Expected (input, output) tuple")
                self.assertEqual(len(data[0]), 1, "Expected one inputs")
                self.assertEqual(len(data[1]), 1, "Expected one outputs")
                self.assertTrue('img' in data[0])
                self.assertTrue('gt' in data[1])
                self.assertTupleEqual(data[0]['img'].shape, (1, 28, 28))
                self.assertTupleEqual(data[1]['gt'].shape, (1,))

            check(train_data)
            check(val_data)
        clear_session()


class TestTutorialTrain(unittest.TestCase):
    def setUp(self) -> None:
        clear_session()

    def test_single_train_iter(self):
        single_train_iter(self, TutorialScenario, get_default_scenario_params())
        clear_session()

    def test_resume_training(self):
        resume_training(self, TutorialScenario, get_default_scenario_params())
        clear_session()

    def test_lav(self):
        lav_test_case(self, TutorialScenario, get_default_scenario_params())
        clear_session()

    def test_warmstart(self):
        warmstart_training_test_case(self, TutorialScenario, get_default_scenario_params())
        clear_session()


if __name__ == '__main__':
    unittest.main()
