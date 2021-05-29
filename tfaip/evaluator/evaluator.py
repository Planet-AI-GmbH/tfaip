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
"""Implementation of a Evaluator

that can be used to define additional metrics computed during LAV.
"""
from typing import Dict, TypeVar, Generic, Type, NoReturn

from tfaip import Sample
from tfaip import EvaluatorParams
from tfaip.util.generic_meta import CollectGenericTypes
from tfaip.util.tfaipargparse import post_init
from tfaip.util.typing import AnyNumpy

TP = TypeVar("TP", bound=EvaluatorParams)


class EvaluatorBase(Generic[TP], metaclass=CollectGenericTypes):
    """
    An EvaluatorBase allows to implement custom metrics for a scenario in pure python.
    It will be applied after the post_proc pipeline (instead of any other metric).

    Overwrite __enter__ to reset the internal states, update_state to update the metrics, and result to yield the
    results.

    Optionally overwrite EvaluatorParams if needed and pass as Generic type.

    See Also:
        - TrainerParams.lav_every_n
        - LAV
    """

    @classmethod
    def params_cls(cls) -> Type[TP]:
        return cls.__generic_types__["TP"]

    @classmethod
    def default_params(cls) -> TP:
        return cls.params_cls()()

    def __init__(self, params: TP):
        post_init(params)
        self.params = params

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def update_state(self, sample: Sample) -> NoReturn:
        """
        Method called after a sample is processed by the model and post_proc to update the internal state of all metrics
        Args:
            sample: unbatched sample (possibly with paddings!)
        See Also:
            tf.keras.metrics.Metric.update_state
        """
        pass

    def result(self) -> Dict[str, AnyNumpy]:
        """
        Method to return the result of the evaluator as dict
        Returns:
            The metric results
        See Also:
            tf.keras.metrics.Metric.result
        """
        return {}
