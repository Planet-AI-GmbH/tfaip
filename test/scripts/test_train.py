import json
import os
import tempfile
import unittest
from subprocess import check_call


class TestTrainingScript(unittest.TestCase):
    def test_train_tutorial(self):
        check_call(['tfaip-train', 'tutorial',
                    '--trainer_params', 'samples_per_epoch=10', 'epochs=2',
                    '--data_params', 'train_batch_size=2',
                    ])

    def test_resume_train_tutorial(self):
        with tempfile.TemporaryDirectory() as d:
            check_call(['tfaip-train', 'tutorial',
                        '--trainer_params', 'samples_per_epoch=10', 'epochs=1', f'checkpoint_dir={d}',
                        '--data_params', 'train_batch_size=2',
                        ])

            # train one more epoch (training was not cancelled)
            trainer_params = json.load(open(os.path.join(d, 'trainer_params.json'), 'r'))
            trainer_params['epochs'] = 2
            json.dump(trainer_params, open(os.path.join(d, 'trainer_params.json'), 'w'))
            check_call(['tfaip-resume-training', d])
