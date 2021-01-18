from tfaip.base.data.pipeline.dataprocessor import DataProcessor
from tfaip.base.data.pipeline.definitions import Sample


class NormalizeProcessor(DataProcessor):
    """
    Example class to show how to use processors that are run in parallel in the samples in the input pipeline.
    This processor will normalize and center the input sample in the range of [-1, 1] (we know the input is in [0, 255]
    """
    def apply(self, sample: Sample) -> Sample:
        inputs = sample.inputs.copy()
        inputs['img'] = ((inputs['img'] / 255) - 0.5) * 2
        return sample.new_inputs(inputs)
