# Copyright 2020 The tfaip authors. All Rights Reserved.
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
from dataclasses import field, dataclass
from typing import Iterable
import logging

from dataclasses_json import dataclass_json

from tfaip.base.data.pipeline.datapipeline import DataGenerator
from tfaip.base.data.pipeline.definitions import Sample, PipelineMode
from tfaip.base import PredictorParams, DataGeneratorParams
from tfaip.base.imports import ScenarioBase, DataBase
from tfaip.util.argumentparser import dc_meta
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
        queue.put(Sample(targets=s.targets, inputs=s.inputs, outputs=s.outputs, meta=s.meta))

    queue.put(None)
    logger.info(f"Generator thread ended.")


class PredictionGenerator(DataGenerator):
    def __init__(self, mode: PipelineMode, params: PredictionGeneratorParams):
        super(PredictionGenerator, self).__init__(mode, params)
        params.predictor_params.silent = True
        self.params = params
        self.scenario, self.scenario_params = ScenarioBase.from_path(params.model)
        self.data = self.scenario.data_cls()(self.scenario_params.data_params)
        self.predict_pipeline = self.data.get_predict_data(params.generator)

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
