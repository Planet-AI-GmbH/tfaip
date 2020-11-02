from dataclasses import dataclass, field
from typing import List

from dataclasses_json import dataclass_json

from tfaip.util.argument_parser import dc_meta


@dataclass_json
@dataclass
class WarmstartParams:
    model: str = field(default=None, metadata=dc_meta(
        help="Path to the saved model or checkpoint to load the weights from."
    ))

    allow_partial: bool = field(default=False, metadata=dc_meta(
        help="Allow that not all weights can be matched."
    ))
    trim_graph_name: bool = field(default=True, metadata=dc_meta(
        help="Remove the graph name from the loaded model and the target model. This is useful if the model name "
             "changed"
    ))
    rename: List[str] = field(default_factory=list, metadata=dc_meta(
        help="A list of renaming rules to perform on the weights. Format: [FROM->TO,FROM->TO,...]"
    ))

    exclude: str = field(default=None, metadata=dc_meta(
        help="A regex applied on the loaded weights to ignore from loading."
    ))
    include: str = field(default=None, metadata=dc_meta(
        help="A regex applied on the loaded weights to include from loading."
    ))

