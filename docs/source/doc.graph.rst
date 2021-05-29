Graph
=====

Each |tfaip| graph must inherit ``GenericGraphBase``, or its subclass ``GraphBase`` which already encapsulates some optional methods for the graph construction.

GraphBase
---------

An example Multi-Layer-Perceptron (MLP) graph for MNIST is shown in the following code example (excerpt of a Tutorial):

.. code-block:: python

    class TutorialGraph(GraphBase[TutorialModelParams]):
        def __init__(self, params: TutorialModelParams, name='conv', **kwargs):
            super(TutorialGraph, self).__init__(params, name=name, **kwargs)
            # Create all layers
            self.flatten = Flatten()
            self.ff = Dense(128, name='f_ff', activation='relu')
            self.logits = Dense(self._params.n_classes, activation=None, name='classify')

        def build_graph(self, inputs, training=None):
            # Connect all layers and return a dict of the outputs
            rescaled_img = K.cast(inputs['img'], dtype='float32') / 255
            logits = self.logits(self.ff(self.flatten(rescaled_img)))
            pred = K.softmax(logits, axis=-1)
            cls = K.argmax(pred, axis=-1)
            out = {'pred': pred, 'logits': logits, 'class': cls}
            return out


Layers are instantiated in the ``__init__`` function and applied in the ``build_graph`` function which is the sole abstract method of ``GraphBase``


GenericGraphBase
----------------

The ``GenericGraphBase`` class provides more flexibility when creating a graph, including different graphs for training (``build_train_graph``) and prediction (``build_prediction_graph``).
This is for example required to implement a sequence-to-sequence model with a decoder that depends on the model (teacher-forcing during training, decoding during prediction).
Furthermode, the method ``pre_proc_targets`` can be overwritten to apply some preprocessing on the dataset targets that are then fed into the metrics.
