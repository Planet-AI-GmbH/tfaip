import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


def softmax_cross_entropy(onehot_labels, logits, tgt_len, label_smoothing=0.):
    if onehot_labels is None:
        raise ValueError("onehot_labels must not be None.")
    if logits is None:
        raise ValueError("logits must not be None.")

    with tf.name_scope("softmax_cross_entropy_loss") as scope:
        logits = ops.convert_to_tensor(logits)
        onehot_labels = math_ops.cast(onehot_labels, logits.dtype)
        logits.get_shape().assert_is_compatible_with(onehot_labels.get_shape())

        if label_smoothing > 0:
            num_classes = math_ops.cast(
                array_ops.shape(onehot_labels)[-1], logits.dtype)
            smooth_positives = 1.0 - label_smoothing
            smooth_negatives = label_smoothing / num_classes
            onehot_labels = onehot_labels * smooth_positives + smooth_negatives

        onehot_labels = array_ops.stop_gradient(
            onehot_labels, name="labels_stop_gradient")
        losses = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logits, name="xentropy")

        loss_mask = tf.sequence_mask(tf.cast(tgt_len - 1, dtype=tf.int32),
                                     tf.cast(tf.shape(onehot_labels)[1], dtype=tf.int32))
        losses = losses * tf.cast(loss_mask, dtype=tf.float32)
        return tf.reduce_sum(losses) / tf.reduce_sum(tf.cast(loss_mask, dtype=tf.float32))
