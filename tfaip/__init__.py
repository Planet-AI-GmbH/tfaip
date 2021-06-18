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
"""Global imports

The global imports import static classes such as parameters and definitions.
The other classes can be imported from ``tfaip.imports``
"""
from tfaip.version import __version__

# IMPORTANT!!!
# Global imports here must not import tensorflow (it won't crash but it is slow)
# This leads to unnecessary imports in spawned sub-processes
from tfaip.data.databaseparams import DataBaseParams, DataGeneratorParams
from tfaip.data.pipeline.definitions import PipelineMode, Sample, INPUT_PROCESSOR, GENERAL_PROCESSOR, TARGETS_PROCESSOR
from tfaip.device.device_config import DeviceConfigParams
from tfaip.evaluator.params import EvaluatorParams
from tfaip.lav.params import LAVParams
from tfaip.model.modelbaseparams import ModelBaseParams
from tfaip.predict.params import PredictorParams
from tfaip.scenario.scenariobaseparams import ScenarioBaseParams
from tfaip.trainer.scheduler.learningrate_params import LearningRateParams
from tfaip.trainer.params import TrainerParams, TrainerPipelineParams, TrainerPipelineParamsBase
from tfaip.trainer.warmstart.warmstart_params import WarmStartParams
