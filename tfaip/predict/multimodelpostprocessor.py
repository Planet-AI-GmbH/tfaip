from dataclasses import dataclass
from typing import List


from tfaip import Sample
from tfaip.data.pipeline.processor.dataprocessor import DataProcessorParams, MappingDataProcessor
from tfaip.predict.multimodelvoter import MultiModelVoter


@dataclass
class MultiModelPostProcessorParams(DataProcessorParams):
    voter: "MultiModelVoter" = None
    post_processors: List[DataProcessorParams] = None

    @staticmethod
    def cls():
        return MultiModelPostProcessor


class MultiModelPostProcessor(MappingDataProcessor[MultiModelPostProcessorParams]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.voter = self.params.voter
        self.post_processors = self.params.post_processors

    def apply(self, sample: Sample) -> Sample:
        split_result = split(sample)
        r = [list(pp.apply([r]))[0] for r, pp in zip(split_result, self.post_processors)]
        r = join(r)
        r = self.voter.vote(r)
        return r


def split(sample: Sample):
    return [
        Sample(inputs=sample.inputs, outputs=output, targets=sample.targets, meta=sample.meta)
        for output in sample.outputs
    ]


def join(samples: List[Sample]):
    return Sample(
        inputs=samples[0].inputs,
        targets=samples[0].targets,
        outputs=[s.outputs for s in samples],
        meta=[s.meta for s in samples],
    )
