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
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorboard.plugins.pr_curve import metadata
from tensorflow.python.keras.utils.metrics_utils import AUCCurve
from tensorflow.python.ops import math_ops


class Curve(tf.keras.metrics.AUC):

    def __init__(self, channel, num_thresholds=200, curve='ROC', name=None, dtype=None,
                 thresholds=None, label_weights=None):
        super().__init__(num_thresholds, curve, 'interpolation', name or curve, dtype, thresholds, False,
                         label_weights)
        self.channel = channel
        self.subset = curve

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super(Curve, self).update_state(
            y_true == self.channel,
            y_pred[:, :, self.channel])
        # tf.unstack(y_pred, axis=-1)[self.channel])

    def result(self):
        # Set `x` and `y` values for the curves based on `curve` config.
        if self.curve == AUCCurve.ROC:
            # false-positive-rate (1-sensitivity)
            x = math_ops.div_no_nan(self.false_positives,
                                        self.false_positives + self.true_negatives)
            # recall/sensitivity
            y = math_ops.div_no_nan(self.true_positives,
                                    self.true_positives + self.false_negatives)
        elif self.curve == AUCCurve.PR:
            # recall
            x = math_ops.div_no_nan(self.true_positives,
                                    self.true_positives + self.false_negatives)
            # precision
            y = math_ops.div_no_nan(self.true_positives,
                                    self.true_positives + self.false_positives)
        else:
            raise Exception("unknown curve")

        return _create_tensor_summary(
            self.name,
            self.subset,
            true_positive_counts=self.true_positives,
            false_positive_counts=self.false_positives,
            true_negative_counts=self.true_negatives,
            false_negative_counts=self.false_negatives,
            precision=y,
            recall=x,
            num_thresholds=self.num_thresholds,
            description=None,
        )


def _create_tensor_summary(
        name,
        subset,
        true_positive_counts,
        false_positive_counts,
        true_negative_counts,
        false_negative_counts,
        precision,
        recall,
        num_thresholds=None,
        description=None,
        collections=None,
):
    """A private helper method for generating a tensor summary.

    We use a helper method instead of having `op` directly call `raw_data_op`
    to prevent the scope of `raw_data_op` from being embedded within `op`.

    Arguments are the same as for raw_data_op.

    Returns:
      A tensor summary that collects data for PR curves.
    """

    # Store the number of thresholds within the summary metadata because
    # that value is constant for all pr curve summaries with the same tag.
    summary_metadata = metadata.create_summary_metadata(
        display_name=name,
        description=description or "",
        num_thresholds=num_thresholds,
    )

    # Store values within a tensor. We store them in the order:
    # true positives, false positives, true negatives, false
    # negatives, precision, and recall.
    combined_data = tf.stack(
        [
            tf.cast(true_positive_counts, tf.float32),
            tf.cast(false_positive_counts, tf.float32),
            tf.cast(true_negative_counts, tf.float32),
            tf.cast(false_negative_counts, tf.float32),
            tf.cast(precision, tf.float32),
            tf.cast(recall, tf.float32),
        ]
    )
    return tf.compat.v1.summary.tensor_summary(
        name=name if not subset else (subset + '/' + name),
        tensor=combined_data,
        collections=collections,
        summary_metadata=summary_metadata,
    )
