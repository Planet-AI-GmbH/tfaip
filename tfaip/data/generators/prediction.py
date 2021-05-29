# Copyright 2021 The tfaip authors. All Rights Reserved.
#
# This file is part of tfaip.
#
# tfaip is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# tfaip is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# tfaip. If not, see http://www.gnu.org/licenses/.
# ==============================================================================
"""Definition of a DataGenerator that generates data based on the prediction of a model"""
import logging
import threading
from dataclasses import field, dataclass
from multiprocessing import Queue
from typing import Iterable

from paiargparse import pai_meta, pai_dataclass

from tfaip import PredictorParams, DataGeneratorParams
from tfaip import Sample, PipelineMode
from tfaip.data.data import DataBase
from tfaip.data.pipeline.datagenerator import DataGenerator
from tfaip.scenario.scenariobase import ScenarioBase

logger = logging.getLogger(__name__)


@pai_dataclass
@dataclass
class PredictionGeneratorParams(DataGeneratorParams):
    model: str = field(default=None, metadata=pai_meta(required=True))
    predictor_params: PredictorParams = field(default_factory=PredictorParams)
    generator: DataGeneratorParams = field(default_factory=DataGeneratorParams)

    @staticmethod
    def cls():
        return PredictionGenerator


def generator(params: PredictionGeneratorParams, data: DataBase, scenario: ScenarioBase, queue: Queue):
    # This function is called in a separate thread.
    # Load the predictor (thus the model) and predict on the generator params of the predictor
    # Write the results to the output queue
    logger.info(f"Loading generator model from {params.model} in separate thread")
    predictor = scenario.predictor_cls()(params.predictor_params, data)
    predictor.set_model(params.model + "/serve")
    for s in predictor.predict(params.generator):
        queue.put(Sample(targets=s.targets, inputs=s.inputs, outputs=s.outputs, meta=s.meta))

    queue.put(None)
    logger.info("Generator thread ended.")


class PredictionGenerator(DataGenerator[PredictionGeneratorParams]):
    """
    This DataGenerator implementation generates data by the output of another model, i.e. the prediction of a
    other network is the input for the actual data pipeline.
    """

    def __init__(self, mode: PipelineMode, params: PredictionGeneratorParams):
        super().__init__(mode, params)
        params.predictor_params.silent = True
        self.scenario, self.scenario_params = ScenarioBase.from_path(params.model)
        self.data = self.scenario.data_cls()(self.scenario_params.data)
        self.predict_pipeline = self.data.predict_data(params.generator)

    def __len__(self):
        return len(self.predict_pipeline.create_data_generator())

    def generate(self) -> Iterable[Sample]:
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
