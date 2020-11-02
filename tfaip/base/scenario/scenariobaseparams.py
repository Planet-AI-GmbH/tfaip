from dataclasses import dataclass, field
from typing import List, Dict

from dataclasses_json import dataclass_json, config, LetterCase

from tfaip.base.data.data_base_params import DataBaseParams
from tfaip.base.model import ModelBaseParams
from tfaip.util.argument_parser import dc_meta


@dataclass_json
@dataclass
class ScenarioBaseParams:
    """
    Define the global params of a scenario
    contains model_params and data_params of the model

    NOTE: add @dataclass_json and @dataclass annotations to inherited class
    """
    debug_graph_construction: bool = field(default=False, metadata=dc_meta(
        help="Build the graph in pure eager mode to debug the graph construction on real data"
    ))
    debug_graph_n_examples: int = field(default=1, metadata=dc_meta(
        help="number of examples to take from the validation set for debugging, -1 = all"
    ))

    print_eval_limit: int = field(default=10, metadata=dc_meta(
        help="Number of evaluation examples to print per evaluation, use -1 to print all"
    ))

    model_params: ModelBaseParams = field(default_factory=lambda: ModelBaseParams())
    data_params: DataBaseParams = field(default_factory=lambda: DataBaseParams())

    scenario_base_path_: str = None
    scenario_module_: str = None
    id_: str = None


@dataclass
class NetConfigNodeSpec:
    shape: List[str]
    dtype: str
    node_frozen: str = field(metadata=config(letter_case=LetterCase.CAMEL))
    node_serve: str = field(metadata=config(letter_case=LetterCase.CAMEL))


@dataclass_json
@dataclass
class NetConfigParamsBase:
    id_model: str = field(metadata=config(letter_case=LetterCase.CAMEL))
    id_frozen: str = field(metadata=config(letter_case=LetterCase.CAMEL))
    id_serve: str = field(metadata=config(letter_case=LetterCase.CAMEL))
    in_nodes: Dict[str, NetConfigNodeSpec] = field(metadata=config(letter_case=LetterCase.CAMEL))
    out_nodes: Dict[str, NetConfigNodeSpec] = field(metadata=config(letter_case=LetterCase.CAMEL))
    tf_version: str = field(metadata=config(letter_case=LetterCase.CAMEL))
