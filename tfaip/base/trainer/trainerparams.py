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
from dataclasses import dataclass, field
from typing import Optional

from dataclasses_json import dataclass_json

from tfaip.base.scenario import ScenarioBaseParams
from tfaip.base.device_config import DeviceConfigParams
from tfaip.base.trainer.callbacks.export_best import ExportBestState
from tfaip.base.trainer.scheduler.learningrate_params import LearningRateParams
from tfaip.base.trainer.warmstart.warmstart_params import WarmstartParams
from tfaip.util.argument_parser import dc_meta
from tfaip.util.versioning import get_commit_hash


@dataclass_json
@dataclass
class OptimizerParams:
    optimizer: str = field(default='Adam', metadata=dc_meta(
        help="The keras optimizer to use."
    ))
    clip_grad: float = field(default=0, metadata=dc_meta(
        help="Gradient clipping. If == 0 -> disabled, > 0: global norm, < 0: local norm"
    ))


@dataclass_json
@dataclass
class TrainerParams:
    epochs: int = field(default=100, metadata=dc_meta(
        help="The number of training epochs."
    ))
    current_epoch: int = field(default=0, metadata=dc_meta(
        help="The epoch to start with. Usually 0, but can be overwritten for resume training."
    ))
    samples_per_epoch: int = field(default=1000, metadata=dc_meta(
        help="The number of samples (not batches!) to process per epoch."
    ))
    train_accum_steps: int = field(default=1, metadata=dc_meta(
        help="Artificially increase the batch size by accumulating the gradients of n_steps(=batches) before applying "
             "them. This factor has to be multiplied with data_params.train_batch_size to compute the 'actual' batch "
             "size"
    ))
    tf_cpp_min_log_level: int = field(default=2, metadata=dc_meta(
        help="The log level for tensorflow cpp code."
    ))
    force_eager: bool = field(default=False, metadata=dc_meta(
        help="Activate eager execution of the graph. See also --scenario_params debug_graph_construction"
    ))
    skip_model_load_test: bool = field(default=False, metadata=dc_meta(
        help="By default, the trainer checks initially whether the prediction model can be saved and loaded. This may "
             "take some time. Thus for debugging you should skip this by setting it to True"
    ))
    test_every_n: int = field(default=1, metadata=dc_meta(
        help="Rate at which to test the model on the validation data (--data_params validation_list)"
    ))
    lav_every_n: int = field(default=0, metadata=dc_meta(
        help="Rate at which to LAV the model during training (similar to test, however on the actual prediction model)."
             "LAV uses --data_params lav_lists"
    ))
    checkpoint_dir: Optional[str] = field(default=None, metadata=dc_meta(
        help="Dictionary to use to write checkpoints, logging files, and export of best and last model."
             "You should provide this."
    ))
    write_checkpoints: bool = field(default=True, metadata=dc_meta(
        help="Write checkpoints to checkpoint_dir during training. Checkpoints are obligatory if you want support to"
             "resume the training (see tfaip-resume-training script)"
    ))
    export_best: Optional[bool] = field(default=None, metadata=dc_meta(
        help="Continuously export the best model during testing to checkpoint_dir/best."
    ))
    export_final: bool = field(default=True, metadata=dc_meta(
        help="Export the final model after training to checkpoint_dir/export."
    ))
    no_train_scope: str = field(default=None, metadata=dc_meta(
        help="Regex to match with layer names to exclude from training, i.e. the weights of these layers won't receive "
             "updates"
    ))
    calc_ema: bool = field(default=False, metadata=dc_meta(
        help="Calculate ema weights. These weights are exported as best or final (prediction model)"
    ))
    random_seed: Optional[int] = field(default=None, metadata=dc_meta(
        help="Random seed for all random generators. Use this to obtain reproducible results (at least on CPU)"
    ))
    profile: bool = field(default=False, metadata=dc_meta(
        help="Enable profiling for tensorboard, profiling batch 10 to 20, initial setup:"
             "pip install -U tensorboard_plugin_profile"
             "LD_LIBRARY_PATH=:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
             'options nvidia "NVreg_RestrictProfilingToAdminUsers=0" to /etc/modprobe.d/nvidia-kernel-common.conf'
             'reboot system'
    ))
    device_params: DeviceConfigParams = field(default_factory=lambda: DeviceConfigParams(), metadata=dc_meta(
        help="Parameters to setup the devices such as GPUs and multi GPU training."
    ))
    optimizer_params: OptimizerParams = field(default_factory=lambda: OptimizerParams(), metadata=dc_meta(
        help="Optimization parameters (e.g. selection of the optimizer)"
    ))
    learning_rate_params: LearningRateParams = field(default_factory=lambda: LearningRateParams(), metadata=dc_meta(
        help="Learning rate, and scheduling parameters."
    ))
    scenario_params: ScenarioBaseParams = field(default_factory=lambda: ScenarioBaseParams(), metadata=dc_meta(
        help="Parameters of the scenario."
    ))
    warmstart_params: WarmstartParams = field(default_factory=lambda: WarmstartParams(), metadata=dc_meta(
        help="Parameters to specify parameters to load before training (e.g. warmstart or finetuning)"
    ))


    # Logging of the best state during training, required for resume training
    export_best_state_: ExportBestState = field(default_factory=lambda: ExportBestState())
    commit_id_: str = field(default_factory=get_commit_hash)
