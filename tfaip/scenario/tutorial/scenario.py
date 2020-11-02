from typing import Type

from tfaip.base.data.data import DataBase
from tfaip.base.scenario import ScenarioBase, ScenarioBaseParams
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from tfaip.base.model import ModelBase
from tfaip.scenario.tutorial.data import Data
from tfaip.scenario.tutorial.model import TutorialModel


@dataclass_json
@dataclass
class ScenarioParams(ScenarioBaseParams):
    pass


class TutorialScenario(ScenarioBase):
    @classmethod
    def model_cls(cls) -> Type['ModelBase']:
        return TutorialModel

    @classmethod
    def data_cls(cls) -> Type['DataBase']:
        return Data

    @staticmethod
    def get_params_cls() -> Type[ScenarioBaseParams]:
        return ScenarioParams

    def __init__(self, params: ScenarioParams):
        super().__init__(params)
