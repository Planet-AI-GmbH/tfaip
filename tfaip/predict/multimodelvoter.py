from abc import ABC, abstractmethod

from tfaip import Sample


class MultiModelVoter(ABC):
    """Class that defines how to vote a Sample produced by multiple predictors (MultiModelPredictor)"""

    @abstractmethod
    def vote(self, sample: Sample) -> Sample:
        """Vote a sample

        Note, this might be run in parallel, so the resulting Sample must be serializable

        Args:
            sample: The multi-sample (sample.outputs is a list of individual predictions)

        Returns:
            A normal sample with the expected structure of sample.outputs
        """
        raise NotImplementedError

    def finalize_sample(self, sample: Sample) -> Sample:
        """Finalize a sample

        This function is called after the call of vote but sequentially, use this to modify non-serializable data
        """
        return sample
