Graph
=====

The ``GraphBase`` class must be implemented to define the actual network architecture of a scenario.
The sole parameter are the :ref:`ModelBaseParams<tfaip.model:ModelBaseParams>`, a graph is constructed in ``ModelBase.create_graph``.

``GraphBase`` extends ``keras.layers.Layer``, hence an implementation of ``call`` is obligatory to join the layers declared in ``__init__``.

An example Multi-Layer-Perceptron (MLP) graph for MNIST is shown in the following code example (excerpt of a Tutorial):

.. code-block:: python

    class TutorialGraph(GraphBase[TutorialModelParams]):
        def __init__(self, params: TutorialModelParams, name='conv', **kwargs):
            super(TutorialGraph, self).__init__(params, name=name, **kwargs)
            # Create all layers
            self.flatten = Flatten()
            self.ff = Dense(128, name='f_ff', activation='relu')
            self.logits = Dense(self._params.n_classes, activation=None, name='classify')

        def call(self, inputs, **kwargs):
            # Connect all layers and return a dict of the outputs
            rescaled_img = K.cast(inputs['img'], dtype='float32') / 255
            logits = self.logits(self.ff(self.flatten(rescaled_img)))
            pred = K.softmax(logits, axis=-1)
            cls = K.argmax(pred, axis=-1)
            out = {'pred': pred, 'logits': logits, 'class': cls}
            return out
