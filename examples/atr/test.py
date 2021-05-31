import unittest

from examples.atr.scenario import ATRScenario
from tensorflow.python.keras.backend import clear_session

from test.util.training import single_train_iter


class ATRTestScenario(ATRScenario):
    @classmethod
    def default_trainer_params(cls):
        params = super().default_trainer_params()
        return params


class TestATR(unittest.TestCase):
    scenario = ATRTestScenario

    def tearDown(self) -> None:
        clear_session()

    def test_single_train_iter(self):
        single_train_iter(self, self.scenario, debug=False)
