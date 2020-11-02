import logging

logger = logging.getLogger(__name__)


def set_global_random_seed(n):
    if n is None:
        return

    logger.info(f"Setting all random seed to {n}")
    import os
    os.environ['PYTHONHASHSEED'] = str(n)
    from numpy.random import seed
    seed(n + 1)
    import tensorflow as tf
    tf.random.set_seed(n + 2)
    import random
    random.seed(n + 3)
