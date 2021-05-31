import unittest

from tensorflow.python.keras.backend import clear_session

from examples.text.finetuningbert.scenario import FTBertScenario
from test.util.training import single_train_iter


class FTBertTestScenario(FTBertScenario):
    @classmethod
    def default_trainer_params(cls):
        params = super().default_trainer_params()
        return params


class TestFineTuningBert(unittest.TestCase):
    scenario = FTBertTestScenario

    def tearDown(self) -> None:
        clear_session()

    def test_single_train_iter(self):
        single_train_iter(self, self.scenario, debug=False)
