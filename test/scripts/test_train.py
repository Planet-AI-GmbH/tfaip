# Copyright 2020 The tfaip authors. All Rights Reserved.
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
