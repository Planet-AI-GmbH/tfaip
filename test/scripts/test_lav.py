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
