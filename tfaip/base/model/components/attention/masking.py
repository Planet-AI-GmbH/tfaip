import tensorflow as tf

def _create_image_padding_mask(seq_len):
    """
    :param seq_len: int32 [batch_size]
    :return:
    """
    padding_mask = tf.sequence_mask(seq_len, tf.reduce_max(seq_len), dtype=tf.int32)
    # (batch_size, max_len) -- > (batch_size, 1, 1, max_len)
    # add extra dimensions to add the padding
    # to the attention logits.
    return tf.cast(padding_mask, dtype=tf.float32)

def _create_padding_mask(seq_len):
    """
    :param seq_len: int32 [batch_size]
    :return:
    """
    padding_mask = tf.sequence_mask(seq_len, tf.reduce_max(seq_len), dtype=tf.int32)
    # (batch_size, max_len) -- > (batch_size, 1, 1, max_len)
    # add extra dimensions to add the padding
    # to the attention logits.
    padding_mask = tf.expand_dims(padding_mask, axis=1)
    padding_mask = tf.expand_dims(padding_mask, axis=1)
    return tf.cast(1 - padding_mask, dtype=tf.float32)


def _create_look_ahead_mask(seq_len):
    """
    Look ahead mask decoder step N is only allowed decoder outputs emitted prior to N
    :param seq_len: int32 scalar
    :return:
    """
    mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

    return tf.cast(mask, dtype=tf.float32)  # (seq_len, seq_len)
