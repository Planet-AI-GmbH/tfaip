import unittest
from subprocess import check_call
import tempfile
import os


class TestLAVScript(unittest.TestCase):
    def test_lav_tutorial(self):
        with tempfile.TemporaryDirectory() as d:
            check_call(['tfaip-train', 'tutorial',
                        '--trainer_params', 'samples_per_epoch=10', 'epochs=1', f'checkpoint_dir={d}',
                        '--data_params', 'train_batch_size=2',
                        ])
            check_call(['tfaip-lav',
                        '--export_dir', os.path.join(d, 'best'),
                        '--data_params', 'val_limit=10',
                        ])
            check_call(['tfaip-lav',
                        '--export_dir', os.path.join(d, 'best'),
                        '--data_params', 'val_limit=10',
                        '--run_eagerly',
                        '--dump', os.path.join(d, 'dump.pkl'),
                        ])
