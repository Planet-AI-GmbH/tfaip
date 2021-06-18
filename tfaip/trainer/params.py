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
"""Definition of the TrainerParams and the TrainerPipelineParamsBase"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union, TypeVar, Generic, Iterable, TYPE_CHECKING

from paiargparse import pai_meta, pai_dataclass

from tfaip.data.databaseparams import DataGeneratorParams, DataPipelineParams
from tfaip import PipelineMode
from tfaip import DeviceConfigParams
from tfaip import ScenarioBaseParams
from tfaip import LearningRateParams
from tfaip.trainer.warmstart.warmstart_params import WarmStartParams
from tfaip.trainer.callbacks.earlystopping.params import EarlyStoppingParams
from tfaip.trainer.optimizer import DEFAULT_OPTIMIZERS
from tfaip.trainer.optimizer.optimizers import AdamOptimizer, OptimizerParams
from tfaip.trainer.scheduler import DEFAULT_SCHEDULES, ExponentialDecayParams
from tfaip.util.generic_meta import CollectGenericTypes, ReplaceDefaultDataClassFieldsMeta

if TYPE_CHECKING:
    from tfaip.data.pipeline.datapipeline import DataPipeline
    from tfaip.data.data import DataBase

TDataGeneratorTrain = TypeVar("TDataGeneratorTrain", bound=DataGeneratorParams)
TDataGeneratorVal = TypeVar("TDataGeneratorVal", bound=DataGeneratorParams)


@pai_dataclass
@dataclass
class TrainerPipelines:
    train: DataPipelineParams = field(
        default_factory=lambda: DataPipelineParams(mode=PipelineMode.TRAINING),
        metadata=pai_meta(fix_dc=True, mode="flat"),
    )
    val: DataPipelineParams = field(
        default_factory=lambda: DataPipelineParams(mode=PipelineMode.EVALUATION),
        metadata=pai_meta(fix_dc=True, mode="flat"),
    )

    def __post_init__(self):
        self.train.mode = PipelineMode.TRAINING
        self.val.mode = PipelineMode.EVALUATION


class TrainerPipelineParamsBaseMeta(CollectGenericTypes):
    pass


@pai_dataclass
@dataclass
class TrainerPipelineParamsBase(
    Generic[TDataGeneratorTrain, TDataGeneratorVal], ABC, metaclass=TrainerPipelineParamsBaseMeta
):
    """Definition of the training pipeline inputs.

    Specify the DataGeneratorParams for Training and Validation as Generaics.
    """

    setup: TrainerPipelines = field(default_factory=TrainerPipelines, metadata=pai_meta(fix_dc=True))

    @classmethod
    def train_cls(cls):
        return cls.__generic_types__[TDataGeneratorTrain.__name__]

    @classmethod
    def val_cls(cls):
        return cls.__generic_types__[TDataGeneratorVal.__name__]

    @abstractmethod
    def train_gen(self) -> TDataGeneratorTrain:
        raise NotImplementedError

    @abstractmethod
    def val_gen(self) -> Optional[TDataGeneratorVal]:
        raise NotImplementedError

    def lav_gen(self) -> Iterable[TDataGeneratorVal]:
        return [self.val_gen()]

    def train_data(self, data: "DataBase") -> "DataPipeline":
        return data.get_or_create_pipeline(self.setup.train, self.train_gen())

    def val_data(self, data: "DataBase") -> "DataPipeline":
        return data.get_or_create_pipeline(self.setup.val, self.val_gen())

    def lav_data(self, data: "DataBase") -> Iterable["DataPipeline"]:
        return (data.create_pipeline(self.setup.val, p) for p in self.lav_gen())

    def __post_init__(self):
        self.setup.__post_init__()


class TrainerPipelineParamsMeta(ReplaceDefaultDataClassFieldsMeta, TrainerPipelineParamsBaseMeta):
    def __new__(mcs, *args, **kwargs):
        return super().__new__(mcs, *args, field_names=["train", "val"], **kwargs)


@pai_dataclass
@dataclass
class TrainerPipelineParams(
    TrainerPipelineParamsBase[TDataGeneratorTrain, TDataGeneratorVal], metaclass=TrainerPipelineParamsMeta
):
    train: TDataGeneratorTrain = field(default_factory=DataGeneratorParams, metadata=pai_meta(mode="flat"))
    val: TDataGeneratorVal = field(default_factory=DataGeneratorParams, metadata=pai_meta(mode="flat"))

    def train_gen(self) -> TDataGeneratorTrain:
        return self.train

    def val_gen(self) -> TDataGeneratorVal:
        return self.val


TTrainerPipelineParams = TypeVar("TTrainerPipelineParams", bound=TrainerPipelineParamsBase)
TScenarioParams = TypeVar("TScenarioParams", bound=ScenarioBaseParams)


class TrainerParamsMeta(ReplaceDefaultDataClassFieldsMeta):
    """Meta class for the trainer params

    The class will automatically replace the defaults of the scenario on gen field with the actual class passed to
    the Generic.
    """

    def __new__(mcs, *args, **kwargs):
        return super().__new__(mcs, *args, field_names=["scenario", "gen"], **kwargs)


@pai_dataclass
@dataclass
class TrainerParams(Generic[TScenarioParams, TTrainerPipelineParams], ABC, metaclass=TrainerParamsMeta):
    """TrainerParams storing hyper-parameters, the ScenarioBaseParams, and the TrainerPipelineParams"""

    epochs: int = field(default=100, metadata=pai_meta(help="The number of training epochs."))
    current_epoch: int = field(
        default=0,
        metadata=pai_meta(help="The epoch to start with. Usually 0, but can be overwritten for resume training."),
    )
    samples_per_epoch: int = field(
        default=-1,
        metadata=pai_meta(
            help="The number of samples (not batches!) to process per epoch. "
            "By default (-1) the size fo the training dataset."
        ),
    )
    scale_epoch_size: float = field(
        default=1,
        metadata=pai_meta(
            help="Multiply the number of samples per epoch by this factor. This is useful when using the dataset size as "
            "samples per epoch (--samples_per_epoch=-1, the default), but if you desire to set it e.g. to the half "
            "dataset size (--scale_epoch_size=0.5)"
        ),
    )
    train_accum_steps: int = field(
        default=1,
        metadata=pai_meta(
            help="Artificially increase the batch size by accumulating the gradients of n_steps(=batches) before applying "
            'them. This factor has to be multiplied with data_params.train_batch_size to compute the "actual" batch '
            "size"
        ),
    )
    progress_bar_mode: int = field(default=1, metadata=pai_meta(help="Verbose level of the progress bar."))
    progbar_delta_time: float = field(
        default=5, metadata=pai_meta(help="If verbose=2 the interval after which to output the current progress")
    )
    tf_cpp_min_log_level: int = field(default=2, metadata=pai_meta(help="The log level for tensorflow cpp code."))
    force_eager: bool = field(default=False, metadata=pai_meta(help="Activate eager execution of the graph."))
    skip_model_load_test: bool = field(
        default=False,
        metadata=pai_meta(
            help="By default, the trainer checks initially whether the prediction model can be saved and loaded. This may "
            "take some time. Thus for debugging you should skip this by setting it to True"
        ),
    )
    val_every_n: int = field(
        default=1,
        metadata=pai_meta(
            help="Rate at which to test the model on the validation data (--data_params validation_list)"
        ),
    )
    lav_every_n: int = field(
        default=0,
        metadata=pai_meta(
            help="Rate at which to LAV the model during training (similar to test, however on the actual prediction model)."
            "LAV uses --data_params lav_lists"
        ),
    )
    lav_silent: bool = field(default=True, metadata=pai_meta(help="Do not print the predictions of lav."))
    lav_min_epoch: int = field(
        default=0,
        metadata=pai_meta(
            help="The epoch must be at least this value to run. This can be handy if lav takes very long in early epochs "
            "e.g. in S2S-models"
        ),
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata=pai_meta(
            help="Dictionary to use to write checkpoints, logging files, and export of best and last model.",
            required=True,
        ),
    )
    write_checkpoints: bool = field(
        default=True,
        metadata=pai_meta(
            help="Write checkpoints to output_dir during training. Checkpoints are obligatory if you want support to"
            "resume the training (see tfaip-resume-training script)"
        ),
    )
    export_best: Optional[bool] = field(
        default=None, metadata=pai_meta(help="Continuously export the best model during testing to output_dir/best.")
    )
    export_final: bool = field(
        default=True, metadata=pai_meta(help="Export the final model after training to output_dir/export.")
    )
    no_train_scope: Optional[str] = field(
        default=None,
        metadata=pai_meta(
            help="Regex to match with layer names to exclude from training, i.e. the weights of these layers will not "
            "receive updates"
        ),
    )
    ema_decay: float = field(
        default=0.0,
        metadata=pai_meta(
            help="Calculate ema weights by decaying the current training weights with the given factor. "
            "These weights are exported as best or final (prediction model). 0.0 means OFF, "
            "greater zero uses this value directly, less than zero calculates ema decay value dynamically. "
            "Values greater equals 1 are not supported."
        ),
    )
    random_seed: Optional[int] = field(
        default=None,
        metadata=pai_meta(
            help="Random seed for all random generators. Use this to obtain reproducible results (at least on CPU)"
        ),
    )
    profile: bool = field(
        default=False,
        metadata=pai_meta(
            help="Enable profiling for tensorboard, profiling batch 10 to 20, initial setup:"
            "pip install -U tensorboard_plugin_profile"
            "LD_LIBRARY_PATH=:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
            'options nvidia "NVreg_RestrictProfilingToAdminUsers=0" to /etc/modprobe.d/nvidia-kernel-common.conf'
            "reboot system"
        ),
    )
    device: DeviceConfigParams = field(
        default_factory=DeviceConfigParams,
        metadata=pai_meta(mode="flat", help="Parameters to setup the devices such as GPUs and multi GPU training."),
    )
    optimizer: OptimizerParams = field(
        default_factory=AdamOptimizer,
        metadata=pai_meta(
            mode="flat",
            help="Optimization parameters (e.g. selection of the optimizer)",
            choices=DEFAULT_OPTIMIZERS,
        ),
    )
    learning_rate: LearningRateParams = field(
        default_factory=ExponentialDecayParams,
        metadata=pai_meta(
            mode="flat",
            help="Learning rate, and scheduling parameters.",
            choices=DEFAULT_SCHEDULES,
        ),
    )
    scenario: TScenarioParams = field(
        default_factory=ScenarioBaseParams, metadata=pai_meta(mode="flat", help="Parameters of the scenario.")
    )
    warmstart: WarmStartParams = field(
        default_factory=WarmStartParams,
        metadata=pai_meta(
            mode="flat", help="Parameters to specify parameters to load before training (e.g. warmstart or finetuning)"
        ),
    )
    early_stopping: EarlyStoppingParams = field(
        default_factory=EarlyStoppingParams, metadata=pai_meta(mode="flat", help="Parameters to define early stopping.")
    )

    # Pipeline setup, Training has a train and val pipeline
    gen: TTrainerPipelineParams = field(
        default_factory=TrainerPipelineParams,
        metadata=pai_meta(help="Parameters that setup the data generators (i.e. the input data)."),
    )

    # Additional params
    saved_checkpoint_sub_dir: str = field(
        default="",
        metadata=pai_meta(mode="ignore", help="Formatted checkpoint_sub_dir of the actual epoch (no formatters)"),
    )
    checkpoint_sub_dir: str = field(
        default="", metadata=pai_meta(mode="ignore", help="`filepath` may contain placeholders such as `{epoch:02d}`")
    )
    checkpoint_save_freq: Union[str, int] = field(
        default="epoch", metadata=pai_meta(mode="ignore", help="or after this many epochs")
    )

    def __post_init__(self):
        pass
