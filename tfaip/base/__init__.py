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

# IMPORTANT!!!
# Global imports here must not import tensorflow (it won't crash but it is slow)
# This leads to unnecessary imports in spawned sub-processes

from tfaip.base.device_config import DeviceConfigParams
from tfaip.base.data.databaseparams import DataBaseParams, DataGeneratorParams
from tfaip.base.data.listfile.listfiledataparams import ListFileDataParams, ListsFilePipelineParams, ListFilePipelineParams
from tfaip.base.evaluator.params import EvaluatorParams
from tfaip.base.lav.params import LAVParams
from tfaip.base.model.modelbaseparams import ModelBaseParams
from tfaip.base.predict.params import PredictorParams
from tfaip.base.scenario.scenariobaseparams import ScenarioBaseParams
from tfaip.base.trainer.params import TrainerParams, OptimizerParams, EarlyStoppingParams, LearningRateParams, WarmstartParams
