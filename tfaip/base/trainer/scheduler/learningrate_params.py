from inspect import isclass
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from tfaip.util.argument_parser import dc_meta
from tfaip.util.enum import StrEnum
# Import all schedules so that they are available in globals()
from tfaip.base.trainer.scheduler.learningrate import *

Schedules = StrEnum('Schedules',
                    {k: k for k, v in globals().items() if isclass(v) and issubclass(v, LearningRateSchedule) and v != LearningRateSchedule})


@dataclass_json
@dataclass
class LearningRateParams:
    type: Schedules = field(default=Schedules.ExpDecay, metadata=dc_meta(
        help="Learning rate decay type."
    ))
    lr: float = field(default=0.001, metadata=dc_meta(
        help="The learning rate."
    ))
    learning_circle: int = field(default=3, metadata=dc_meta(
        help="(type dependent) The number of epochs with a flat constant learning rate"
    ))
    lr_decay_rate: float = field(default=0.99, metadata=dc_meta(
        help="(type dependent) The exponential decay factor"
    ))
    decay_fraction: float = field(default=0.1, metadata=dc_meta(
        help="(type dependent) Alpha value of cosine decay"
    ))
    final_epochs: int = field(default=50, metadata=dc_meta(
        help="(type dependent) Number of final epochs with a steep decline in the learning rate"
    ))
    step_function: bool = field(default=True, metadata=dc_meta(
        help="(type dependent) Step function of exponential decay,"
    ))

    warmup_epochs: int = field(default=10, metadata=dc_meta(
        help="(type dependent) Number of epochs with an increasing learning rate."
    ))
    warmup_factor: int = field(default=10, metadata=dc_meta(
        help="(type dependent) Factor from which to start warmup learning (lr/fac)"
    ))
    constant_epochs: int = field(default=10, metadata=dc_meta(
        help="(Type dependent) Number of constant epochs before starting lr decay"
    ))

    # Updated during training to support loading and resuming
    steps_per_epoch_: int = -1
    epochs_: int = -1

    def create(self):
        if self.epochs_ < 0:
            raise ValueError("Epochs not specified.")

        return globals()[self.type.value](self)
