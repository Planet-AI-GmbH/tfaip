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
from tfaip import DataBaseParams, Sample
from tfaip.imports import MultiModelPredictor, MultiModelVoter


class TutorialVoter(MultiModelVoter):
    """
    This MultiModelVoter performs a majority vote of several predictions.
    Alternatively, one could sum up all probabilities of the classes and pick the argmax.
    """

    def vote(self, sample: Sample) -> Sample:
        # sample.outputs is a list of the output of each model
        # just do a majority voting
        counts = {}
        for output in sample.outputs:
            p = output["class"]
            counts[p] = counts.get(p, 0) + 1

        voted = max(counts.items(), key=lambda kv: kv[1])[0]
        return sample.new_outputs({"class": voted})


class TutorialMultiModelPredictor(MultiModelPredictor):
    """
    Tutorial class for a MultiModelPredictor to show how to implement a voting mechanism to vote the output of
    multiple models.
    """

    def create_voter(self, data_params: "DataBaseParams") -> MultiModelVoter:
        # Create an instance of the voter
        return TutorialVoter()
