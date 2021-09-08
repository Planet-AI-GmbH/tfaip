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
