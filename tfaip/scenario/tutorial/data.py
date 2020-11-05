from dataclasses import dataclass, field
import logging
import tensorflow as tf
import tensorflow.keras as keras
from dataclasses_json import dataclass_json

from tfaip.base.data.data import DataBaseParams, DataBase
from tfaip.util.argument_parser import dc_meta

logger = logging.getLogger(__name__)


@dataclass_json
@dataclass
class DataParams(DataBaseParams):
    dataset: str = field(default='fashion_mnist', metadata=dc_meta(
        help="The dataset to select."
    ))


class Data(DataBase):
    @staticmethod
    def get_params_cls():
        return DataParams

    def __init__(self, params: DataParams):
        super().__init__(params)
        self._params = params

        # This tutorial is not based on lists, so just set them as dummy since the values are checked for non empty
        self._params.train_lists = ['NOT_REQUIRED']
        self._params.val_list = 'NOT_REQUIRED'

        dataset = getattr(keras.datasets, self._params.dataset)
        self._train, self._test = dataset.load_data()

    def _get_train_data(self):
        def group(img, gt):
            return {'img': img}, {'gt': gt}
        return tf.data.Dataset.from_tensor_slices(self._train).repeat().map(group)

    def _get_val_data(self, val_list):
        def group(img, gt):
            return {'img': img}, {'gt': gt}
        return tf.data.Dataset.from_tensor_slices(self._test).map(group)

    def _input_layer_specs(self):
        return {'img': tf.TensorSpec(shape=(28, 28), dtype='uint8')}

    def _target_layer_specs(self):
        return {'gt': tf.TensorSpec(shape=[], dtype='uint8')}
