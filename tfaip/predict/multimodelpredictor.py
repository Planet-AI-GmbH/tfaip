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
"""Definition of the MultiModelPredictor"""
import json
import os
from abc import abstractmethod, ABC
from typing import Union, TYPE_CHECKING, List, Iterable, Type

from tensorflow import keras

from tfaip import Sample, PipelineMode
from tfaip import PredictorParams
from tfaip.data.pipeline.datapipeline import DataPipeline
from tfaip.predict.predictorbase import PredictorBase
from tfaip.util.json_helper import TFAIPJsonDecoder
from tfaip.util.multiprocessing.parallelmap import tqdm_wrapper

if TYPE_CHECKING:
    from tfaip.scenario.scenariobase import ScenarioBase
    from tfaip.data.data import DataBase
    from tfaip.data.databaseparams import DataBaseParams


class MultiModelVoter(ABC):
    """
    Class that defines how to vote a Sample produced by multiple predictors (MultiModelPredictor)
    """

    @abstractmethod
    def vote(self, sample: Sample) -> Sample:
        """
        Vote a sample

        Args:
            sample: The multi-sample (sample.outputs is a list of individual predictions)

        Returns:
            A normal sample with the expected structure of sample.outputs
        """
        raise NotImplementedError


class MultiModelPredictor(PredictorBase):
    """
    The MultiModelPredictor supports to apply multiple models on the same data and then apply a MultiModelVoter to
    fuse their results into one.

    Create either with Scenario.create_multi_predictor or MultiModelPredictor.from_paths.

    To use a MultiModelPredictor the create_voter method must be implemented that defines how to vote the multiple
    individual results. Note that the pre processing pipeline of all predictors must be identical.
    The post_processing pipeline of the provided data will be applied

    See Also:
        PredictorBase
    """

    @classmethod
    def from_paths(
        cls,
        paths: List[str],
        params: PredictorParams,
        scenario: Type["ScenarioBase"],
        use_first_params=False,
        model_paths: List[str] = None,
        models: List[keras.models.Model] = None,
        predictor_args=None,
    ) -> "MultiModelPredictor":
        """
        Create a MultiModePredictor. The data of the first model (in paths) will be used as the defining scenario and
        data (i.e. the post-processing). All data pre-procs must be identical.

        Args:
            paths: paths to the scenario_params (see ScenarioBase.params_from_path)
            params: PredictorParams
            scenario: Type of the ScenarioBase
            use_first_params: Only False is supportet ATM.
            model_paths: Paths to the actual models saved dirs (optional), by default based on paths
            models: Already instantiated models (optional), by default created based on paths
            predictor_args: Additional args for instantiating the Predictior (that are not part of PredictorParams)

        Returns:
            An instantiated and ready to use MultiModelPredictor
        """
        predictor_args = predictor_args or {}
        if len(paths) == 0:
            raise ValueError("No paths provided")
        # load scenario params from path, check that the pre proc and post proc pipelines are identical
        scenarios = [scenario.params_from_path(path) for path in paths]
        data_params = scenarios[0].data
        if not use_first_params:
            for p in scenarios[1:]:
                if p.data.pre_proc != data_params.pre_proc:
                    raise ValueError(f"The preprocessors differ {p.data.pre_proc} and {data_params.pre_proc}")

        predictor = cls(params=params, data=scenario.data_cls()(data_params), **predictor_args)

        if not models:
            model_paths = model_paths or [os.path.join(model, "serve") for model in paths]
            models = [
                keras.models.load_model(model, compile=False, custom_objects=scenario.model_cls().all_custom_objects())
                for model in model_paths
            ]
        predictor.set_models(models, [scenario.data_cls()(s.data) for s in scenarios])
        return predictor

    def __init__(self, params: PredictorParams, data: "DataBase"):
        super().__init__(params, data)
        self._datas: List["DataBase"] = []
        self.params.pipeline.mode = PipelineMode.EVALUATION if self.params.include_targets else PipelineMode.PREDICTION

    @abstractmethod
    def create_voter(self, data_params: "DataBaseParams") -> MultiModelVoter:
        raise NotImplementedError

    def set_models(self, models: List[Union[str, keras.Model]], datas: List["DataBase"]):
        # Set the multiple models
        # the Function will create one large joined keras Model that applies all given models in parallel
        # and thus produces a list of dictionaries as output.
        assert len(models) == len(datas), "The number of models and DataBases must coincide"
        models = [self._load_model(model) for model in models]

        for i, model in enumerate(models):
            model._name = f"{i}_{model.name}"  # Forced renaming, pylint: disable=protected-access

        class JoinedModel(keras.Model):
            def call(self, inputs, training=None, mask=None):
                return [model(inputs) for model in models]

            def get_config(self):
                raise NotImplementedError

        self._keras_model = JoinedModel()
        self._datas = datas

    @property
    def datas(self) -> List["DataBase"]:
        return self._datas

    def _unwrap_batch(self, inputs, targets, outputs, meta) -> Iterable[Sample]:
        try:
            batch_size = next(iter(inputs.values())).shape[0]
        except StopIteration as e:
            raise ValueError(f"Empty inputs {inputs}") from e
        for i in range(batch_size):
            un_batched_outputs = [{k: v[i] for k, v in output.items()} for output in outputs]
            un_batched_inputs = {k: v[i] for k, v in inputs.items()}
            un_batched_targets = {k: v[i] for k, v in targets.items()}
            un_batched_meta = {k: v[i] for k, v in meta.items()}
            parsed_meta = json.loads(un_batched_meta["meta"][0].decode("utf-8"), cls=TFAIPJsonDecoder)
            sample = Sample(
                inputs=un_batched_inputs, outputs=un_batched_outputs, targets=un_batched_targets, meta=parsed_meta
            )

            yield sample

    def _print_prediction(self, sample: Sample, print_fn):
        for i, output in enumerate(sample.outputs):
            print_fn(
                f"\n     PREDICTION {i:02d}:\n"
                + "\n".join(
                    [f"        {k}: mean = {v.mean()}, max = {v.max()}, min = {v.min()}" for k, v in output.items()]
                )
            )

    def predict_pipeline(self, pipeline: DataPipeline) -> Iterable[Sample]:
        # The pipeline prediction is overwritten since the batched results must be splitted into a single non-batched
        # sample that comprised all individual prediction outputs as outputs.
        voter = self.create_voter(self._data.params)
        post_processors = [
            d.get_or_create_pipeline(self.params.pipeline, pipeline.generator_params).create_output_pipeline()
            for d in self._datas
        ]
        with pipeline as rd:

            def split(sample: Sample):
                return [
                    Sample(inputs=sample.inputs, outputs=output, targets=sample.targets, meta=sample.meta)
                    for output in sample.outputs
                ]

            def join(samples: List[Sample]):
                return Sample(
                    inputs=samples[0].inputs,
                    targets=samples[0].targets,
                    outputs=[s.outputs for s in samples],
                    meta=[s.meta for s in samples],
                )

            results = tqdm_wrapper(
                self.predict_dataset(rd.input_dataset()),
                progress_bar=self._params.progress_bar,
                desc="Prediction",
                total=len(rd),
            )
            split_results = map(split, results)
            for split_result in split_results:
                r = [list(pp.apply([r]))[0] for r, pp in zip(split_result, post_processors)]
                r = join(r)
                yield voter.vote(r)
