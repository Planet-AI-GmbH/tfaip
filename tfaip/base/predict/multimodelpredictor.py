import json
import os
from abc import abstractmethod, ABC
from typing import Union, TYPE_CHECKING, List, Iterable, Type

from tensorflow import keras

from tfaip.base.data.pipeline.datapipeline import DataPipeline
from tfaip.base.data.pipeline.definitions import InputOutputSample
from tfaip.base.predict.predictorbase import PredictorBase, PredictorParams
from tfaip.util.multiprocessing.parallelmap import tqdm_wrapper

if TYPE_CHECKING:
    from tfaip.base.scenario import ScenarioBase
    from tfaip.base.data.data import DataBase
    from tfaip.base.data.data_base_params import DataBaseParams


class MultiModelVoter(ABC):
    @abstractmethod
    def vote(self, sample: InputOutputSample) -> InputOutputSample:
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
        for i, model in enumerate(models):
            model._name = f"{i}_{model.name}"  # Forced renaming
        outputs = [model(inputs) for model in models]
        self._keras_model = keras.models.Model(inputs=inputs, outputs=(inputs, outputs))
        self._datas = datas

    @property
    def datas(self) -> List['DataBase']:
        return self._datas

    def _unwrap_batch(self, inputs, outputs) -> Iterable:
        batch_size = next(iter(inputs.values())).shape[0]
        for i in range(batch_size):
            un_batched_outputs = [{k: v[i] for k, v in output.items()} for output in outputs]
            un_batched_inputs = {k: v[i] for k, v in inputs.items()}
            sample = InputOutputSample(un_batched_inputs, un_batched_outputs)

            yield sample

    def _print_prediction(self, sample: InputOutputSample, print_fn):
        for i, output in enumerate(sample.outputs):
            print_fn(f"\n     PREDICTION {i:02d}:\n" + "\n".join([f'        {k}: mean = {v.mean()}, max = {v.max()}, min = {v.min()}' for k, v in output.items()]))

    def predict_pipeline(self, pipeline: DataPipeline) -> Iterable[InputOutputSample]:
        voter = self.create_voter(self._data.params())
        post_processors = [d.get_predict_data(pipeline.generator_params).create_output_pipeline() for d in self._datas]
        with pipeline as rd:
            def split(sample: InputOutputSample):
                return [InputOutputSample(sample.inputs, output, sample.meta) for output in sample.outputs]

            def join(samples: List[InputOutputSample]):
                return InputOutputSample(samples[0].inputs, [s.targets for s in samples], [s.meta for s in samples])

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
