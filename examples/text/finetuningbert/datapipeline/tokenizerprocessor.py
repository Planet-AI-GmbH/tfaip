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
from dataclasses import dataclass
from typing import Type

import numpy as np
from paiargparse import pai_dataclass
from tfaip import Sample
from tfaip.data.pipeline.processor.dataprocessor import DataProcessorParams, MappingDataProcessor, DataProcessorBase

from examples.text.finetuningbert.params import Keys

logger = logging.getLogger(__name__)


@pai_dataclass
@dataclass
class TokenizerProcessorParams(DataProcessorParams):
    model_name: str = ""

    @staticmethod
    def cls() -> Type["DataProcessorBase"]:
        return TokenizerProcessor


class TokenizerProcessor(MappingDataProcessor[TokenizerProcessorParams]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.params.model_name) > 0
        # load the tokenizer, local import for parallel processing support
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.params.model_name)
        logger.info(f"Loaded Tokenizer with a vocab size of {self.tokenizer.vocab_size}")

    def apply(self, sample: Sample) -> Sample:
        def encode_sentences(sentence1, sentence2):
            tokens1 = list(self.tokenizer.tokenize(sentence1)) + [self.tokenizer.sep_token]
            tokens2 = list(self.tokenizer.tokenize(sentence2)) + [self.tokenizer.sep_token]
            return [self.tokenizer.cls_token] + tokens1 + tokens2, [0] + [0] * len(tokens1) + [1] * len(tokens2)

        word_ids, type_ids = encode_sentences(sample.inputs[Keys.InputSentence1], sample.inputs[Keys.InputSentence2])
        word_ids = self.tokenizer.convert_tokens_to_ids(word_ids)
        return sample.new_inputs(
            {
                Keys.InputWordIds: word_ids,
                Keys.InputMask: np.full(fill_value=1, shape=[len(word_ids)], dtype=np.int32),
                Keys.InputTypeIds: np.asarray(type_ids, dtype=np.int32),
            }
        )
