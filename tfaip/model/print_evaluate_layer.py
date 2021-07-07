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
"""Definition of the PrintEvaluateLayer"""
import json
import logging
from typing import TYPE_CHECKING, List, NamedTuple, Any

import tensorflow as tf

from tfaip import PipelineMode, Sample
from tfaip.util.json_helper import TFAIPJsonDecoder

if TYPE_CHECKING:
    from tfaip.scenario.scenariobase import ScenarioBase

logger = logging.getLogger(__name__)


class PrintEvaluateLayerInput(NamedTuple):
    inputs: Any
    outputs: Any
    targets: Any
    meta: Any


class PrintEvaluateLayer(tf.keras.layers.Layer):
    """
    Purpose: call model.print_evaluate during evaluation to view current prediction results
    Method:  Function must be injected to any node (similar to tf.print, here to the first loss), done in ScenarioBase
             All outputs will be converted to numpy arrays, then model.print_evaluate is called
    """

    def __init__(self, scenario: "ScenarioBase", limit=-1, name="print_evaluate_layer"):
        super().__init__(name=name, trainable=False)
        self.scenario = scenario
        self.limit_reached = False
        self.limit = limit
        self._still_allowed = tf.Variable(self.limit, trainable=False, name="print_limit")
        if scenario is None:
            logger.warning(
                "You are instantiating a PrintEvaluateLayer without a scenario. This is not supported. "
                "Probably you loaded the keras model with keras.models.load instead of reinstantiating "
                "the full graph (e.g. using tfaip-resume-training). This will not result in an error, but "
                "no outputs will be generated."
            )

        data = self.scenario.data
        val_pipeline = data.pipeline_by_mode(PipelineMode.EVALUATION)
        self._post_proc = val_pipeline.create_output_pipeline()

    def get_config(self):
        # Implement this to get rid of warning.
        cfg = super().get_config()
        cfg["scenario"] = None  # No scenario
        return cfg

    def operation(self, inputs, training=None):
        # IDEA:
        # Counter (self._still_allowed) counts how many outputs still to process during training=False
        # during training: counter is reset to self.limit
        # during prediction: call print on batch, and reduce counter by batch size
        assert isinstance(inputs, PrintEvaluateLayerInput)

        if self.scenario is None:
            # No scenario set. This should usually not be the case, only when loading a training Graph
            # Instead recreate full graph.
            return tf.constant(0)

        def print_op():
            # Only print if we didnt reach the limit
            batch_size = tf.shape(tf.nest.flatten(inputs)[0])[0]

            def real_print():
                # tf function requires a list as input, hack is to pack everything into a list and storing the key
                # names, and unpacking in the actual print function
                # also the "training" state must be packed...
                return tf.py_function(self.run_print, pack(inputs), Tout=[], name="print")

            a = tf.cond(tf.greater(self._still_allowed, 0), real_print, tf.no_op)
            with tf.control_dependencies([a]):
                b = self._still_allowed.assign_sub(batch_size)
            return tf.group([a, b])

        def reset_op():
            with tf.control_dependencies([self._still_allowed.assign(self.limit)]):
                return tf.no_op()

        if isinstance(training, bool):
            op = reset_op() if training else print_op()
        else:
            op = tf.cond(tf.convert_to_tensor(training), reset_op, print_op)

        with tf.control_dependencies([op]):
            return tf.constant(0)  # return a constant (0) since the output of the PEL is used as a dummy loss

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        return self.operation(inputs, training)

    def run_print(self, *args):
        i, o, t, m = unpack(list(args))  # values to pass though, pylint: disable=unbalanced-tuple-unpacking
        # Copy current value of still allowed to set the correct limit batch wise
        still_allowed = self._still_allowed.numpy()
        if still_allowed != 0:
            # take an arbitrary output, this defines the batch size
            pel_key = next(iter(o.keys()))
            if len(o[pel_key].shape) == 0:
                raise ValueError("Loss must have shape of batch size, but got a single value instead")
            batch_size = o[pel_key].shape[0]
            # test if numpy calls possible, else its graph building mode
            in_np = {k: v.numpy() for k, v in i.items()}
            out_np = {k: v.numpy() for k, v in o.items()}
            target_np = {k: v.numpy() for k, v in t.items()}
            meta_np = {k: v.numpy() for k, v in m.items()}
            if still_allowed == self.limit:
                logger.info(f'Printing evaluation results of {"all" if self.limit == -1 else self.limit} instances')
            for batch in range(batch_size):
                inputs = {k: v[batch] for k, v in in_np.items()}
                outputs = {k: v[batch] for k, v in out_np.items()}
                targets = {k: v[batch] for k, v in target_np.items()}
                meta = {k: v[batch] for k, v in meta_np.items()}
                meta = json.loads(meta["meta"][0], cls=TFAIPJsonDecoder)
                sample = Sample(inputs=inputs.copy(), outputs=outputs, targets=targets, meta=meta)
                post_proc_sample = self._post_proc.apply_on_sample(sample)
                self.scenario.model.print_evaluate(post_proc_sample, self.scenario.data, print_fn=logger.info)
                still_allowed -= 1
                if still_allowed == 0:
                    break
        return tf.no_op()


def pack(dicts: List[dict]) -> list:
    out_keys = []
    out = [out_keys]
    for i, d in enumerate(dicts):
        if not isinstance(d, dict):
            out_keys.append(f"_{-1:03d}_")
            out.append(d)
        else:
            for k, v in d.items():
                out_keys.append(f"_{i:03d}_{k}")
                out.append(v)

    return out


def unpack(lists: list) -> List[dict]:
    out_keys = lists[0]
    dicts = []
    for key, l in zip(out_keys, lists[1:]):
        key = key.numpy().decode("utf-8")
        prefix = key[:5]
        real_key = key[5:]
        idx = int(prefix[1:-1])
        if idx < 0:
            dicts.append(l)
            continue

        while idx >= len(dicts):
            dicts.append({})

        d = dicts[idx]
        d[real_key] = l

    return dicts
