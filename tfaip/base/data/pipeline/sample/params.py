from dataclasses import dataclass, field
from typing import List

from dataclasses_json import dataclass_json

from tfaip.base.data.pipeline.definitions import DataProcessorFactoryParams


@dataclass_json
@dataclass
class SamplePipelineParams:
    run_parallel: bool = True
    sample_processors: List[DataProcessorFactoryParams] = field(default_factory=list)
