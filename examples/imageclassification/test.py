import unittest

from examples.imageclassification.scenario import ICScenario
from tensorflow.python.keras.backend import clear_session

from test.util.training import single_train_iter


class ICTestScenario(ICScenario):
    @classmethod
    def default_trainer_params(cls):
        params = super().default_trainer_params()
        return params


class TestIC(unittest.TestCase):
    scenario = ICTestScenario

    def tearDown(self) -> None:
        clear_session()

    def test_single_train_iter(self):
        single_train_iter(self, self.scenario, debug=False)
