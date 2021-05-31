import os
import pathlib
from dataclasses import dataclass

import tensorflow as tf

from paiargparse import pai_dataclass
from tfaip import ScenarioBaseParams
from tfaip.scenario.scenariobase import ScenarioBase

from examples.imageclassification.datapipeline.datagenerator import ICTrainerPipelineParams
from examples.imageclassification.datapipeline.predictiondatagenerator import ICPredictionDataGeneratorParams
from examples.imageclassification.model import ICModelParams
from examples.imageclassification.params import ICDataParams

# download the flowers dataset and extract it
from examples.imageclassification.predictor import ICPredictor

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file("flower_photos", origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)


@pai_dataclass
@dataclass
class ICScenarioParams(ScenarioBaseParams[ICDataParams, ICModelParams]):
    def __post_init__(self):
        self.model.num_classes = len(self.data.classes)


class ICScenario(ScenarioBase[ICScenarioParams, ICTrainerPipelineParams]):
    @classmethod
    def default_trainer_params(cls):
        p = super().default_trainer_params()
        p.gen.dataset_path = data_dir.absolute().as_posix()
        p.gen.setup.train.batch_size = 5
        p.gen.setup.val.batch_size = 5
        p.epochs = 10
        return p

    @classmethod
    def create_trainer(cls, trainer_params, restore=False):
        # setup the classes of the data here, since now we are sure they wont change anymore
        trainer_params.scenario.data.classes = list(os.listdir(trainer_params.gen.dataset_path))
        # this function will also call post_init recursively so that the number of classes are set
        return super().create_trainer(trainer_params, restore)

    @classmethod
    def predict_generator_params_cls(cls):
        # During prediction, the default ICDataGeneratorParams can not be used because the require a class (target)
        # which is not available during training. Return an other DataGenerator that only acts on unlabeled images
        return ICPredictionDataGeneratorParams

    @classmethod
    def predictor_cls(cls):
        # For a nicer formatting of the output
        return ICPredictor
