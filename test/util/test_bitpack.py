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
#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Gundram Leifert"
__copyright__ = "Copyright 2021, Planet AI GmbH"
__credits__ = ["Gundram Leifert"]
__email__ = "gundram.leifert@planet-ai.de"

import numpy as np
import tensorflow as tf

from tfaip.util.packbits import packbits, unpackbits, packsize


def test_unpackbits():
    for length in [s for s in range(1, 30)]:
        print(f"test length={length}")
        res = np.random.random([5, 6, 7, length]) > 0.5
        res = res.astype(np.uint8)
        packed_np = packbits(res)
        packed_tf = tf.convert_to_tensor(packed_np)

        assert packed_np.shape[-1] == packsize(length)
        assert packed_tf.shape[-1] == packsize(length)

        unpacked_tf = unpackbits(packed_tf, length)
        unpacked_np = unpacked_tf.numpy()
        np.testing.assert_equal(unpacked_np, res)

        unpacked_np_ = unpackbits(packed_np, length)
        np.testing.assert_equal(unpacked_np_, res)
    # print(unpacked_np)
