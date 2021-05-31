import os
from dataclasses import dataclass

from paiargparse import pai_dataclass
from tfaip import ScenarioBaseParams
from tfaip.scenario.scenariobase import ScenarioBase

from examples.atr.model import ATRModelParams
from examples.atr.params import ATRDataParams, ATRTrainerPipelineParams

this_dir = os.path.dirname(os.path.realpath(__file__))


@pai_dataclass
@dataclass
class ATRScenarioParams(ScenarioBaseParams[ATRDataParams, ATRModelParams]):
    def __post_init__(self):
        self.model.num_classes = len(self.data.codec) + 1  # +1 for blank


class ATRScenario(ScenarioBase[ATRScenarioParams, ATRTrainerPipelineParams]):
    @classmethod
    def default_trainer_params(cls):
        p = super().default_trainer_params()
        default_data_dir = os.path.join(this_dir, "workingdir", "uw3_50lines")
        p.gen.train.image_files = [os.path.join(default_data_dir, "train", "*.png")]
        p.gen.val.image_files = [os.path.join(default_data_dir, "train", "*.png")]
        p.gen.setup.train.batch_size = 5
        p.gen.setup.val.batch_size = 5
        p.samples_per_epoch = 1024
        return p
