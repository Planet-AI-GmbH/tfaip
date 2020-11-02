import tensorflow.keras as keras


class MaxPool2D(keras.layers.MaxPool2D):
    def __init__(self, **kwargs):
        super(MaxPool2D, self).__init__(**kwargs)

    def call(self, input, mask=None, **kwargs):
        if mask is not None:
            assert(len(input.get_shape()) == len(mask.get_shape()))
            input -= (1 - mask) * 1e10

        return super(MaxPool2D, self).call(input)
