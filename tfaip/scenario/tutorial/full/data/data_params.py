from dataclasses import dataclass, field

from dataclasses_json import dataclass_json

from tfaip.base import DataBaseParams
from tfaip.util.argumentparser import dc_meta


@dataclass_json
@dataclass
class DataParams(DataBaseParams):
    dataset: str = field(default='mnist', metadata=dc_meta(
        help="The dataset to select (chose also fashion_mnist)."
    ))
