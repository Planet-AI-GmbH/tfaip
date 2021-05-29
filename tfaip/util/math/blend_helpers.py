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
import logging
from collections.abc import Iterator, Iterable
from numbers import Number
from random import Random
from typing import List

from tfaip.util.math.key_helper import KeyHelper
from tfaip.util.math.iter_helpers import ListIterablor


logger = logging.getLogger(__name__)


class BlendIterablor(Iterator, Iterable):
    def __init__(self, ingredients: List[ListIterablor], mixing_ratios: List[float], random: Random):
        """
        :param ingredients: iterablors
        :param mixing_ratios: mixing ratios
        :param random: random number generator
        :type random: Random
        """
        assert isinstance(random, Random)
        self._random = random
        self.ingredients = ingredients
        assert isinstance(ingredients, list)
        assert isinstance(mixing_ratios, list)
        assert len(ingredients) == len(mixing_ratios)
        for ingredient in ingredients:
            assert isinstance(ingredient, Iterator)
        for mixing_ratio in mixing_ratios:
            assert isinstance(mixing_ratio, Number)
            assert mixing_ratio >= 0
        ratio_sum = float(sum(mixing_ratios))
        assert ratio_sum > 0.0
        key = 0.0
        self._cumulative_mri_dict = {}
        num = len(ingredients)
        # remove 0.0 ratios
        cnt = 0
        for ingredient, mixing_ratio in zip(ingredients, mixing_ratios):
            if mixing_ratio > 0:
                key = key + float(mixing_ratio) / ratio_sum
                cnt = cnt + 1
                if cnt == num:
                    key = 1.0
                try:
                    ingredient.shuffle(self._random)
                except AttributeError:
                    logger.warning(f'ingredient {type(ingredient)}: missing method "shuffle(random)" - skipping')
                self._cumulative_mri_dict[key] = ingredient
        self._key_helper = KeyHelper(self._cumulative_mri_dict)

    def __len__(self):
        return sum(map(len, self.ingredients))

    def __iter__(self):
        return self

    def __next__(self):
        r = self._random.random()
        key = self._key_helper.get_key(r)
        return self._cumulative_mri_dict[key].next()
