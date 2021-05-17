Model
=====

The model glues together several parts that define the setup of the neural network, e.g. the :ref:`loss<doc.model:Loss>` or the :ref:`metrics<doc.model:Metric>`.

The implementation of the model requires to override the base class :ref:`ModelBase<tfaip.model:ModelBase>`
and its parameters :ref:`ModelBaseParams<tfaip.model:ModelBaseParams>` (see the following example of the tutorial):

.. code-block:: python

    @pai_dataclass
    @dataclass
    class TutorialModelParams(ModelBaseParams):
        n_classes: int = field(default=10, metadata=pai_meta(
            help="The number of classes (depends on the selected dataset)"
        ))

        @staticmethod
        def cls():
            return TutorialModel

        def graph_cls(self):
            from examples.tutorial.min.graphs import TutorialGraph
            return TutorialGraph

    class TutorialModel(ModelBase[TutorialModelParams]):
        pass

Parameter Overrides
-------------------

The implementation of the ``ModelBaseParams`` require to override ``cls()`` and ``graph_cls`` to return the class type of the actual model and :ref:`graph<doc.graph:Graph>`.

Loss
----

The loss function defines the optimization target of the model.
There are two ways to define a loss: loss using a ``keras.losses.Loss``, or a loss using a Tensor as output.
Multiple losses can be :ref:`weighted<doc.model:loss weight>`.
The output-values of each loss (and the weighted loss) will be displayed in the console and in the :ref:`Tensorboard<doc.model:tensorboard>`.

Overwrite ``_loss`` and return a dictionary of losses where the key is the (display) label of the metric and the value is the Tensor-valued loss.

To use a ``keras.losses.Loss``, instantiate the loss in the ``__init__``-function and call it in ``_loss``.
Alternatively, return any scalar-valued Tensor.

.. code-block:: python

    def __init(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scc_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True, name='keras_loss')

    def _loss(self, inputs, targets, outputs) -> Dict[str, AnyTensor]:
         return {
            'keras_loss': self.scc_loss(targets['gt'], outputs['logits']),  # either call a keras.Loss
            'raw_loss': tf.keras.losses.sparse_categorical_crossentropy(targets['gt'], outputs['logits'], from_logits=True),  # or add a raw loss
        }

Loss Weight
~~~~~~~~~~~

If multiple losses are defined, the ``_loss_weights`` function can be implemented to return weights for the losses.
Here both upper losses are weighted with a factor of 0.5.
If not implemented, each loss is weighted by a factor of 1.

.. code-block:: python

    def _loss_weights(self) -> Optional[Dict[str, float]]:
        return {'keras_loss': 0.5, 'extended_loss': 0.5}

Metric
------

Similar to the loss, a model defines its metrics.
The output-values of each metric will be displayed in the console and in the :ref:`Tensorboard<doc.model:Tensorboard>`.
All metrics are computed on both the training and validation data, except the :ref:`pure Python<doc.model:pure-python metric>` one which is solely computed on the validation set.

Overwrite ``_metric`` and return a list of called ``keras.metric.Metric``.
The ``name`` of the metric is used for display.

.. code-block:: python

    def __init(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acc_metric = keras.metrics.Accuracy(name='acc')

    def _metric(self, inputs, targets, outputs):
        return [self.acc_metric(targets['gt'], outputs['class'])]

Custom metrics must implement ``keras.metrics.Metric``.
It is also possible to compute the actual value of the metric as Tensor beforehand and wrap it with a ``keras.metrics.Mean``.

Pure-Python Metric
~~~~~~~~~~~~~~~~~~

Pure python metrics are not defined with the Model but instead in the :ref:`Evaluator<doc.scenario:evaluator>`.
They provide a maximum of flexibility since they are computed during :ref:`load and validate<doc.training:extended>` in pure Python.

Logging the best model
----------------------

During :ref:`training<doc.training:training>` the best model will be tracked and automatically exported as "best".
The best model is determined by a models ``_best_logging_settings`` which is by default the minimum loss since every model provides this information.
If you want to track the best model for example by a metric, overwrite this function.
For instance, if a model defines a :ref:`metric<doc.model:Metric>` ``"acc"``, use

.. code-block:: python

    def _best_logging_settings(self):
        return "max", "acc"

The first return value is either ``"max"`` or ``"min"`` while the second argument is the name of a metric or loss.


Output during validation
------------------------

During validation the first few examples are passed to a ``Model``'s ``_print_evaluate`` function which can be used to display the current state of training in a human-readable form.
For MNIST-training this could be the target class and the prediction probabilities, e.g.:

.. code-block:: python

    def _print_evaluate(self, sample: Sample, data, print_fn=print):
        outputs, targets = sample.outputs, sample.targets
        correct = outputs['class'] == targets['gt']
        print_fn(f"PRED/GT: {outputs['class']}{'==' if correct else '!='}{targets['gt']} (p = {outputs['pred'][outputs['class']]})")

Note that a sample is already un-batched.
This function can also access to the ``data``-class if a mapping (e.g. a codec) must be applied.

Tensorboard
-----------

During training, the output of the loss and metrics on the training and validation sets is automatically to the Tensorboard.
The data is stored in the ``output_dir`` defined during [training](07_training.md).

In some cases, additional :ref:`arbitrary data<doc.model:arbitrary data>` such as images, or raw data e.g. such as :ref:`PR-curves<doc.model:pr-curves>` shall be written to the Tensorboard.

Arbitrary Data
~~~~~~~~~~~~~~

To add arbitrary additional data to the Tensorboard ensure that the layer adding the data inherits ``TFAIPLayerBase`` which provides a method ``add_tensorboard`` which must be called with a ``TensorboardWriter`` and the ``value``.

The following examples shows how to write the output of a conv-layer as image to the Tensorboard.
The ``TensorboardWriter`` will receive the raw numpy data and call the provided ``func`` (here ``handle``) to process the raw data and write it to the tensorboard.

.. code-block:: python

    def handle(name: str, value: np.ndarray, step: int):
        # Create the image data as numpy array
        b, w, h, c = value.shape
        ax_dims = int(np.ceil(np.sqrt(c)))
        out_conv_v = np.zeros([b, w * ax_dims, h * ax_dims, 1])
        for i in range(c):
            x = i % ax_dims
            y = i // ax_dims
            out_conv_v[:, x * w:(x + 1) * w, y * h:(y + 1) * h, 0] = value[:, :, :, i]

        # Write the image (use 'name_for_tb' and step)
        tf.summary.image(name, out_conv_v, step=step)

    class Layers(TFAIPLayerBase):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.conv_layer = Conv2D(40)
            self.conv_mat_tb_writer = TensorboardWriter(func=handle, dtype='float32', name='conv_mat')

        def call(self, inputs, **kwargs):
            conv_out = self.conv_layer(inputs)
            self.add_tensorboard(self.conv_mat_tb_writer, conv_out)
            return conv_out

PR-curves
~~~~~~~~~

If a metric (e.g. the PR-curve) returns binary data (already serialized Tensorboard data) it will be automatically written to the Tensorboard.

Exporting additional graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def _export_graphs(self,
                       inputs: Dict[str, tf.Tensor],
                       outputs: Dict[str, tf.Tensor],
                       targets: Dict[str, tf.Tensor],
                       ) -> Dict[str, tf.keras.Model]:
        # Override this function
        del targets  # not required in the default implementation
        return {"default": tf.keras.Model(inputs=inputs, outputs=outputs)}

This function defines the graphs to export.
By default, this is the graph defined by all inputs and all outputs.
Override this function to export a different or additional graphs, e.g., if you want to only export the encoder in an encoder/decoder setup.
Return a Dict with ``label`` and ``keras.models.Model`` to export.

Root-Graph-Construction
-----------------------

The root graph can be overwritten to have full flexibility when creating a graph.
In most cases this is optional.

.. code-block:: python

    @staticmethod
    def root_graph_cls() Type['RootGraph']:
        from tfaip.model.graphbase import RootGraph
        return RootGraph
