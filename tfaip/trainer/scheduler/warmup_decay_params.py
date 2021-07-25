from dataclasses import dataclass, field

from paiargparse import pai_dataclass, pai_meta

from tfaip import LearningRateParams


@pai_dataclass(alt="WarmupDecay")
@dataclass
class WarmupDecayParams(LearningRateParams):
    """Cosine decay with warmup"""

    @staticmethod
    def cls():
        from tfaip.trainer.scheduler.warmup_decay import (
            WarmupDecaySchedule,
        )  # pylint: disable=import-outside-toplevel

        return WarmupDecaySchedule

    warmup_epochs: int = field(default=-1, metadata=pai_meta(help="Number of epochs for linear increase"))
    warmup_steps: int = field(default=-1, metadata=pai_meta(help="Number of epochs for linear increase"))
