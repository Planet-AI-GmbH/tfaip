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
import json
import os
from abc import abstractmethod, ABC
from typing import Union, TYPE_CHECKING, List, Iterable, Type

from tensorflow import keras

from tfaip.base.data.databaseparams import DataGeneratorParams
from tfaip.base.data.pipeline.datapipeline import DataPipeline
from tfaip.base.data.pipeline.definitions import Sample, PipelineMode
from tfaip.base.predict.predictorbase import PredictorBase, PredictorParams
from tfaip.util.multiprocessing.parallelmap import tqdm_wrapper

if TYPE_CHECKING:
    from tfaip.base.scenario import ScenarioBase
    from tfaip.base.data.data import DataBase
    from tfaip.base.data.databaseparams import DataBaseParams


class MultiModelVoter(ABC):
    @abstractmethod
    def vote(self, sample: Sample) -> Sample:
        raise NotImplementedError


class MultiModelPredictor(PredictorBase):
    @classmethod
    def from_paths(cls, paths: List[str], params: PredictorParams, scenario: Type['ScenarioBase'],
                   use_first_params=False, model_paths: List[str] = None, models: List[keras.models.Model] = None, predictor_args={}) -> 'MultiModelPredictor':
        if len(paths) == 0:
            raise ValueError("No paths provided")
        # load scenario params from path, check that the pre proc and post proc pipelines are identical
        scenarios = [scenario.params_from_path(path) for path in paths]
        data_params = scenarios[0].data_params
        if not use_first_params:
            for p in scenarios[1:]:
                if p.data_params.pre_processors_ != data_params.pre_processors_:
                    raise ValueError(f"The preprocessors differ {p.data_params.pre_processors_} and {data_params.pre_processors_}")

        predictor = cls(params=params, data=scenario.data_cls()(data_params), **predictor_args)

        if not models:
            model_paths = model_paths or [os.path.join(model, 'serve') for model in paths]
            models = [keras.models.load_model(model,
                                              compile=False,
                                              custom_objects=scenario.model_cls().get_all_custom_objects())
                      for model in model_paths
                      ]
        predictor.set_models(models, [scenario.data_cls()(s.data_params) for s in scenarios])
        return predictor

    def __init__(self, params: PredictorParams, data: 'DataBase'):
        super(MultiModelPredictor, self).__init__(params, data)
        self._datas: List['DataBase'] = []

    @abstractmethod
    def create_voter(self, data_params: 'DataBaseParams') -> MultiModelVoter:
        raise NotImplementedError

    def set_models(self, models: List[Union[str, keras.Model]], datas: List['DataBase']):
        assert(len(models) == len(datas))
        models = [self._load_model(model, False) for model in models]
        inputs = self._data.create_input_layers()
        outputs = [model(inputs) for model in models]
        for i, model in enumerate(models):
            model._name = f"{i}_{model.name}"  # Forced renaming
        if self._params.include_targets:
            targets = self._data.create_target_as_input_layers()
            joined = {**inputs, **targets}
            self._keras_model = keras.models.Model(inputs=joined, outputs=(inputs, targets, outputs))
        else:
            self._keras_model = keras.models.Model(inputs=inputs, outputs=(inputs, outputs))
        self._datas = datas

    @property
    def datas(self) -> List['DataBase']:
        return self._datas

    def _unwrap_batch(self, inputs, targets, outputs) -> Iterable[Sample]:
        batch_size = next(iter(inputs.values())).shape[0]
        for i in range(batch_size):
            un_batched_outputs = [{k: v[i] for k, v in output.items()} for output in outputs]
            un_batched_inputs = {k: v[i] for k, v in inputs.items()}
            un_batched_targets = {k: v[i] for k, v in targets.items()}
            sample = Sample(inputs=un_batched_inputs, outputs=un_batched_outputs, targets=un_batched_targets)

            yield sample

    def _print_prediction(self, sample: Sample, print_fn):
        for i, output in enumerate(sample.outputs):
            print_fn(f"\n     PREDICTION {i:02d}:\n" + "\n".join([f'        {k}: mean = {v.mean()}, max = {v.max()}, min = {v.min()}' for k, v in output.items()]))

    def predict_pipeline(self, pipeline: DataPipeline) -> Iterable[Sample]:
        voter = self.create_voter(self._data.params())
        pipeline_mode = PipelineMode.Evaluation if self.params.include_targets else PipelineMode.Prediction
        post_processors = [d.get_pipeline(pipeline_mode, pipeline.generator_params).create_output_pipeline() for d in self._datas]
        with pipeline as rd:
            def split(sample: Sample):
                return [Sample(inputs=sample.inputs, outputs=output, targets=sample.targets, meta=sample.meta) for output in sample.outputs]

            def join(samples: List[Sample]):
                return Sample(inputs=samples[0].inputs, targets=samples[0].targets,
                              outputs=[s.outputs for s in samples], meta=[s.meta for s in samples])

            results = tqdm_wrapper(self.predict_database(rd.input_dataset()),
                                   progress_bar=self._params.progress_bar,
                                   desc="Prediction",
                                   total=len(rd),
                                   )
            split_results = map(split, results)
            for split_result in split_results:
                r = [list(pp.apply([r]))[0] for r, pp in zip(split_result, post_processors)]
                r = join(r)
                yield voter.vote(r)
