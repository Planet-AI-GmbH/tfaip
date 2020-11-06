import unittest
from subprocess import check_call
import tempfile
import os


class TestPredictScript(unittest.TestCase):
    def test_predict_tutorial(self):
        with tempfile.TemporaryDirectory() as d:
            check_call(['tfaip-train', 'tutorial',
                        '--trainer_params', 'samples_per_epoch=10', 'epochs=1', f'checkpoint_dir={d}',
                        '--data_params', 'train_batch_size=2',
                        ])
            check_call(['tfaip-predict',
                        '--predict_lists', 'NONE',
                        '--export_dir', os.path.join(d, 'best'),
                        '--data_params', 'val_limit=10',
                        ])
