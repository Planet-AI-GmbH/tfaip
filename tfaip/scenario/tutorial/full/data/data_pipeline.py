import glob

from tfaip.base import ListFilePipelineParams
from tfaip.base.data.pipeline.datapipeline import DataPipeline, DataGenerator, RawDataGenerator
from tfaip.base.data.pipeline.definitions import PipelineMode, Sample
from tfaip.util.imaging.io import load_image_from_img_file


def to_samples(samples):
    return [Sample(inputs={'img': img}, targets={'gt': gt.reshape((1,))}) for img, gt in zip(*samples)]


class TutorialPipeline(DataPipeline):
    def create_data_generator(self) -> DataGenerator:
        if self.mode == PipelineMode.Training:
            return RawDataGenerator(to_samples(self.data.train), self.mode, self.generator_params)
        elif self.mode == PipelineMode.Evaluation:
            return RawDataGenerator(to_samples(self.data.test), self.mode, self.generator_params)
        elif self.mode == PipelineMode.Prediction:
            if isinstance(self.generator_params, ListFilePipelineParams):
                # Instead of loading images to a raw pipeline, you should create a custom preprocessing pipeline
                # That is used during training and prediction
                assert self.generator_params.list, "No images provided"
                return RawDataGenerator(
                    [Sample(inputs={'img': img}) for img in
                     map(load_image_from_img_file, glob.glob(self.generator_params.list))],
                    self.mode, self.generator_params)
            else:
                return RawDataGenerator(to_samples(self.data.test), self.mode, self.generator_params)
        elif self.mode == PipelineMode.Targets:
            return RawDataGenerator(to_samples(self.data.test), self.mode, self.generator_params)
