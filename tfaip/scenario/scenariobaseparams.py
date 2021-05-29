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
"""Definition of the ScenarioBaseParams"""
import importlib
from abc import ABC
from dataclasses import dataclass, field
from typing import List, Dict, Optional, TypeVar, Generic, Type, NoReturn, TYPE_CHECKING

from dataclasses_json import config, LetterCase
from paiargparse import pai_meta, pai_dataclass

from tfaip import __version__
from tfaip.data.databaseparams import DataBaseParams
from tfaip import EvaluatorParams
from tfaip import ModelBaseParams
from tfaip.util.generic_meta import ReplaceDefaultDataClassFieldsMeta
from tfaip.util.versioning import get_commit_hash

if TYPE_CHECKING:
    from tfaip.scenario.scenariobase import ScenarioBase

TDataParams = TypeVar("TDataParams", bound=DataBaseParams)
TModelParams = TypeVar("TModelParams", bound=ModelBaseParams)


class ScenarioBaseParamsMeta(ReplaceDefaultDataClassFieldsMeta):
    def __new__(mcs, *args, **kwargs):
        return super().__new__(mcs, *args, field_names=["data", "model"], **kwargs)


@pai_dataclass
@dataclass
class ScenarioBaseParams(Generic[TDataParams, TModelParams], ABC, metaclass=ScenarioBaseParamsMeta):
    """
    Define the global params of a scenario
    contains model_params and data_params of the model

    NOTE: add @pai_dataclass and @dataclass annotations to inherited class
    """

    @classmethod
    def data_cls(cls) -> Type[TDataParams]:
        """Returns the class of the data params"""
        return cls.__generic_types__[TDataParams.__name__]

    @classmethod
    def model_cls(cls) -> Type[TModelParams]:
        """Returns the class of the model params"""
        return cls.__generic_types__[TModelParams.__name__]

    debug_graph_n_examples: int = field(
        default=1, metadata=pai_meta(help="number of examples to take from the validation set for debugging, -1 = all")
    )

    print_eval_limit: int = field(
        default=10, metadata=pai_meta(help="Number of evaluation examples to print per evaluation, use -1 to print all")
    )

    tensorboard_logger_history_size: int = field(
        default=5,
        metadata=pai_meta(help="Number of instances to store for outputing into tensorboard. Default (last n=5)"),
    )

    export_serve: bool = field(default=True, metadata=pai_meta(help="Export the serving model (saved model format)"))

    model: TModelParams = field(
        default_factory=ModelBaseParams,
        metadata=pai_meta(
            mode="flat",
        ),
    )
    data: TDataParams = field(
        default_factory=DataBaseParams,
        metadata=pai_meta(
            mode="flat",
        ),
    )
    evaluator: EvaluatorParams = field(default_factory=EvaluatorParams, metadata=pai_meta(mode="flat"))

    # Additional export params
    export_net_config: bool = field(default=True, metadata=pai_meta(mode="ignore"))
    net_config_filename: str = field(default="net_config.json", metadata=pai_meta(mode="ignore"))
    default_serve_dir: str = field(default="serve", metadata=pai_meta(mode="ignore"))
    additional_serve_dir: str = field(default="additional", metadata=pai_meta(mode="ignore"))
    trainer_params_filename: str = field(default="trainer_params.json", metadata=pai_meta(mode="ignore"))
    scenario_params_filename: str = field(default="scenario_params.json", metadata=pai_meta(mode="ignore"))

    scenario_base_path: Optional[str] = field(default=None, metadata=pai_meta(mode="ignore"))
    scenario_id: Optional[str] = field(default=None, metadata=pai_meta(mode="ignore"))
    id: Optional[str] = field(default=None, metadata=pai_meta(mode="ignore"))

    tfaip_commit_hash: str = field(default_factory=get_commit_hash, metadata=pai_meta(mode="ignore"))
    tfaip_version: str = field(default=__version__, metadata=pai_meta(mode="ignore"))

    def __post_init__(self) -> NoReturn:
        """
        This function is called after __init__ but also when instantiation the corresponding module.

        Override this for parameter sharing.
        """
        pass

    def cls(self) -> Type["ScenarioBase"]:
        if self.scenario_id is None:
            raise ValueError(
                "Scenario param 'scenario_id' not set. Cannot determine scenario type automatically. "
                "Please override cls() in the scenario params or set 'scenario_id'."
            )
        module, cls = self.scenario_id.split(":")
        return getattr(importlib.import_module(module), cls)

    def create(self) -> "ScenarioBase":
        return self.cls()(self)


@dataclass
class NetConfigNodeSpec:
    shape: List[str]
    dtype: str
    node_serve: str = field(metadata=config(letter_case=LetterCase.CAMEL))


@pai_dataclass
@dataclass
class NetConfigParamsBase:
    id_model: str = field(metadata=config(letter_case=LetterCase.CAMEL))
    id_serve: str = field(metadata=config(letter_case=LetterCase.CAMEL))
    in_nodes: Dict[str, NetConfigNodeSpec] = field(metadata=config(letter_case=LetterCase.CAMEL))
    out_nodes: Dict[str, NetConfigNodeSpec] = field(metadata=config(letter_case=LetterCase.CAMEL))

    tf_version: str = field(metadata=config(letter_case=LetterCase.CAMEL))
    tfaip_commit_hash: str = field(default_factory=get_commit_hash, metadata=config(letter_case=LetterCase.CAMEL))
    tfaip_version: str = field(default=__version__, metadata=config(letter_case=LetterCase.CAMEL))
