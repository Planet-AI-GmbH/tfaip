from tfaip.base.data.pipeline.definitions import Sample
from tfaip.base.predict.multimodelpredictor import MultiModelPredictor, MultiModelVoter


class TutorialVoter(MultiModelVoter):
    def vote(self, sample: Sample) -> Sample:
        # sample.outputs is a list of the output of each model
        # just do a majority voting
        counts = {}
        for output in sample.outputs:
            p = output['class']
            counts[p] = counts.get(p, 0) + 1

        voted = max(counts.items(), key=lambda kv: kv[1])[0]
        return sample.new_outputs({'class': voted})


class TutorialMultiModelPredictor(MultiModelPredictor):
    def create_voter(self, data_params: 'DataBaseParams') -> MultiModelVoter:
        return TutorialVoter()
