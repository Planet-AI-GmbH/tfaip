from dataclasses import field, dataclass
from typing import Iterable
import logging

from dataclasses_json import dataclass_json

from tfaip.base.data.data import DataBase
from tfaip.base.data.data_base_params import DataGeneratorParams
from tfaip.base.data.pipeline.datapipeline import DataGenerator
from tfaip.base.data.pipeline.definitions import InputTargetSample, PipelineMode
from tfaip.base.predict import PredictorParams
from tfaip.base.scenario import ScenarioBase
from tfaip.util.argument_parser import dc_meta
from multiprocessing import Queue
import threading


logger = logging.getLogger(__name__)


@dataclass_json
@dataclass
class PredictionGeneratorParams(DataGeneratorParams):
    model: str = field(default=None, metadata=dc_meta(required=True))
    predictor_params: PredictorParams = field(default_factory=PredictorParams, metadata=dc_meta(arg_mode='ignore'))
    generator: DataGeneratorParams = field(default_factory=DataGeneratorParams, metadata=dc_meta(arg_mode='snake'))


def generator(params: PredictionGeneratorParams, data: DataBase, scenario: ScenarioBase, queue: Queue):
    logger.info(f"Loading generator model from {params.model} in separate thread")
    predictor = scenario.predictor_cls()(params.predictor_params, data)
    predictor.set_model(params.model + '/serve')
    for s in predictor.predict_pipeline(data.get_predict_data(params.generator)):
        queue.put(InputTargetSample(s.targets, s.inputs, s.meta))

    queue.put(None)
    logger.info(f"Generator thread ended.")


class PredictionGenerator(DataGenerator):
    def __init__(self, mode: PipelineMode, params: PredictionGeneratorParams):
        super(PredictionGenerator, self).__init__(mode, params)
        self.params = params
        self.scenario, self.scenario_params = ScenarioBase.from_path(params.model)
        self.data = self.scenario.data_cls()(self.scenario_params.data_params)
        self.predict_pipeline = self.data.get_predict_data(params.generator)

    def __len__(self):
        return len(self.predict_pipeline.create_data_generator())

    def generate(self) -> Iterable[InputTargetSample]:
        # Launch model in a separate thread, else tf.data.Dataset throws some weired bugs
        queue = Queue(maxsize=2 * self.params.batch_size)
        thread = threading.Thread(target=generator, args=(self.params, self.data, self.scenario, queue))
        thread.start()
        while thread.is_alive() or not queue.empty():
            r = queue.get()
            if r is None:
                # Prediction Finished
                break
            yield r

        thread.join()
