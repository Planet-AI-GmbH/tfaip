from dataclasses import dataclass

from paiargparse import pai_dataclass

from examples.text.finetuningbert.datapipeline.gluedata import GlueTrainerPipelineParams
from examples.text.finetuningbert.datapipeline.tokenizerprocessor import TokenizerProcessorParams
from examples.text.finetuningbert.model import FTBertModelParams
from examples.text.finetuningbert.params import FTBertDataParams
from tfaip import ScenarioBaseParams
from tfaip.scenario.scenariobase import ScenarioBase
from tfaip.trainer.scheduler import WarmupCosineDecayParams


@pai_dataclass
@dataclass
class FTBertScenarioParams(ScenarioBaseParams[FTBertDataParams, FTBertModelParams]):
    def __post_init__(self):
        for p in self.data.pre_proc.processors_of_type(TokenizerProcessorParams):
            p.model_name = self.model.model_name


class FTBertScenario(ScenarioBase[FTBertScenarioParams, GlueTrainerPipelineParams]):
    @classmethod
    def default_trainer_params(cls):
        p = super().default_trainer_params()
        p.gen.setup.train.batch_size = 32
        p.gen.setup.val.batch_size = 32
        p.epochs = 3
        p.learning_rate = WarmupCosineDecayParams(lr=2e-5, warmup_epochs=1, warmup_factor=100)
        return p
