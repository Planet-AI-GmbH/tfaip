Model
=====

The model glues together several parts that define the setup of the neural network, e.g. the :ref:`Graph<doc.model:Graph-Construction>`, the :ref:`loss<doc.model:Loss>`, or the :ref:`metrics<doc.model:Metric>`.

The implementation of the model requires to override the base class :ref:`ModelBase<tfaip.model:ModelBase>`
and its parameters :ref:`ModelBaseParams<tfaip.model:ModelBaseParams>` (see the following example of the tutorial):

.. code-block:: python

    @pai_dataclass
    @dataclass
    class ModelParams(ModelBaseParams):
        n_classes: int = field(default=10, metadata=pai_meta(
            help="The number of classes (depends on the selected dataset)"
        ))


    class Model(ModelBase[ModelParams]):
        pass

Graph-Construction
------------------

Override the following to define the :ref:`graph<doc.graph:Graph>`:

.. code-block:: python

    def create_graph(self, params: ModelParams):
        return Graph(params)

Loss
----

The loss function defines the optimization target of the model.
There are two ways to define a loss: loss using a ``keras.losses.Loss`` (see :ref:`Keras Loss<doc.model:keras loss>`), or a loss using a Tensor as output (see :ref:`extended-loss<doc.model:extended loss>`).
Multiple losses can be :ref:`weighted<doc.model:loss weight>`.
The output-values of each loss (and the weighted loss) will be displayed in the console and in the :ref:`Tensorboard<doc.model:tensorboard>`.

Keras Loss
~~~~~~~~~~

A Keras loss is the simplest method to define a loss.
Overwrite ``_loss`` and return a dictionary of losses each being a ``LossDefinition`` tuple of ``target``, ``output``, and ``keras.Loss``.

.. code-block:: python

    def _loss(self) -> Dict[str, LossDefinition]:
        return {'keras_loss': LossDefinition('gt', 'logits', tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))}

The drawback of this approach is that only one target and one output can be accessed to compute the loss.
However, this is satisfied in most cases.

Extended Loss
~~~~~~~~~~~~~

The return value of the loss is a Dictionary of tensors of shape batch-size.
Since all inputs, targets, and outputs can be accessed, arbitrary losses can be defined.

.. code-block:: python

    def _extended_loss(self, inputs_targets, outputs) -> Dict[str, AnyTensor]:
        return {'extended_loss': tf.keras.losses.sparse_categorical_crossentropy(inputs_targets['gt'], outputs['logits'], from_logits=True)}

``inputs_targets`` is the joined dictionary of the inputs and targets coming from the dataset, ``outputs`` hold the outputs of the network.

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
Similar to the loss, there are multiple (here three) different approaches with an increasing flexibility.
The output-values of each metric will be displayed in the console and in the :ref:`Tensorboard<doc.model:Tensorboard>`.
All metrics are computed on both the training and validation data, except the :ref:`pure Python<doc.model:pure-python metric>` one which is solely computed on the validation set.

Keras Metric
~~~~~~~~~~~~

A Keras metric is the simplest method of defining a metric.
Overwrite ``_metric`` and return a dictionary of ``MetricDefinitions`` which is a tuple of the ``target`` and ``output`` tensors which are fed in the actual ``keras.Metric``.
Either pass a custom ``keras.Metric`` or use one out of the box, e.g.:

.. code-block:: python

    def _metric(self):
        return {'acc': MetricDefinition("gt", "class", keras.metrics.Accuracy())}

The drawback of this metric is, that only one target and one output is used but which is the default in most cases.
The advantage is, that keras metrics are flexible by calling ``update_state`` to arbitrarily accumulate the metric values and finally ``result`` to obtain the final value.
You can overwrite ``_sample_weights`` to provide the ``weights`` of a metric batch as third input to ``update_state``.

Extended Metric
~~~~~~~~~~~~~~~

The definition of an extended metric is identical to the definition of the losses: simply return a dict of Tensors.
The final metric is computed by averaging, implement ``_sample_weights`` to define the weighting factors.
See the example in the Tutorial:

.. code-block:: python

    def _extended_metric(self, inputs_targets, outputs):
        return {'acc': tf.keras.metrics.sparse_categorical_accuracy(inputs_targets['gt'], outputs['pred'])}

The drawback of this metric is that you can not correctly compute any metric since the sample weights can not map any scenario (e.g. precision and recall).
The advantage is that an extended metric has access to all ``inputs``, ``targets`` and ``outputs`` and can thus compute metrics that require multiple inputs.

Multi Metric
~~~~~~~~~~~~

``MultiMetrics`` are an _optional_ extension to the standard keras metrics.
They enable to hierarchically compute metrics that are all based on intermediate values, e.g., first compute TP, FP, FN, then compute the derived metrics precision, recall, and F1.
To use implement a ``MultiMetric`` overwrite ``_precomputed_values`` to compute derived tensors of any shape (e.g. dicts).
These tensors will then be passes to the attached child-metrics that are stated upon definition of the ``MultiMetric``, see e.g.:

.. code-block:: python

    def _multi_metric(self) -> Dict[str, MultiMetricDefinition]:
        class MyMultiMetric(MultiMetric):
            def _precompute_values(self, y_true, y_pred, sample_weight):
                # Compute some intermediate values that will be used in the sub metrics
                # Here, the Identity is returned, and applied to the default keras Accuracy metrics (see below)
                return y_true, y_pred, sample_weight

        return {'multi_metric': MultiMetricDefinition('gt', 'class', MyMultiMetric([keras.metrics.Accuracy(name='macc1'), keras.metrics.Accuracy(name='macc2')]))}

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

In some cases, additional data such as :ref:`images<doc.model:images>` or :ref:`PR-curves<doc.model:pr-curves>` shall be written to the Tensorboard.
This is enabled by implementing a ``TensorBoardDataHandler`` that defines which outputs of the models are excluded from the command line and thus only written to teh Tensorboard and
how the data shall be handled:

.. code-block:: python

    def _create_tensorboard_handler(self) -> 'TensorBoardDataHandler':
        class ExampleTBHandler(TensorBoardDataHandler):
            # OVERRIDE
            pass
        return ExampleTBHandler()

In the following, a few examples are provided how to pass a certain type of data to the Tensorboard.

Images
~~~~~~

This tensorboard handler (part of the full-tutorial) shows how to write image data (last batch of validation) to the Tensorboard.
The image is the output of the conv layers.

.. code-block:: python

    def _create_tensorboard_handler(self) -> 'TensorBoardDataHandler':
        class TutorialTBHandler(TensorBoardDataHandler):
            def _outputs_for_tensorboard(self, inputs, outputs) -> Dict[str, AnyTensor]:
                # List the outputs of the model that are used for the Tensorboard
                # Here, access the 'conv_out'
                return {k: v for k, v in outputs.items() if k in ['conv_out']}

            def handle(self, name, name_for_tb, value, step):
                # Override handle to state, that something other than writing a scalar must be performed
                # for a output. Value is the output of the network as numpy array
                if name == 'conv_out':
                    # Create the image data as numpy array
                    b, w, h, c = value.shape
                    ax_dims = int(np.ceil(np.sqrt(c)))
                    out_conv_v = np.zeros([b, w * ax_dims, h * ax_dims, 1])
                    for i in range(c):
                        x = i % ax_dims
                        y = i // ax_dims
                        out_conv_v[:,x*w:(x+1)*w,y*h:(y+1)*h, 0] = value[:,:,:,i]

                    # Write the image (use 'name_for_tb' and step)
                    tf.summary.image(name_for_tb, out_conv_v, step=step)
                else:
                    # The default case, write a scalar
                    super(TutorialTBHandler, self).handle(name, name_for_tb, value, step)

        return TutorialTBHandler()

PR-curves
~~~~~~~~~

To be continued.

Additional overrides
--------------------

The following is a list of other functions that can be overwritten.

Additional Layers
~~~~~~~~~~~~~~~~~

.. code-block:: python

    @classmethod
    def _additional_layers(cls) -> List[keras.layers.Layer]:
        return []

This function shall return the list of all layers that are passed to keras for reconstruction after loading an exported model.
This is to support eager execution during :ref:`LAV<doc.evaluation:load and validate (LAV)>` or :ref:`prediction<doc.prediction:prediction>`).
The default implementation searches all graphs (classes that inherit ``GraphBase``) in either the ``graphs.py`` file or a ``graphs``-package.
Note, it is sufficient to list the top-most layers, usually the base graphs.

Exporting additional graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def _export_graphs(self,
                       inputs: Dict[str, tf.Tensor],
                       outputs: Dict[str, tf.Tensor],
                       targets: Dict[str, tf.Tensor],
                       ) -> List[ExportGraph]:
        return [ExportGraph("default", inputs=inputs, outputs=outputs)]

This function defines the graphs to export.
By default, this is the graph defined by all inputs and all outputs.
Override this function to export a different or additional graphs, e.g., if you want to only export the encoder in an encoder/decoder setup.
The ``ExportGraph`` data-structure expects a name, and the inputs and outputs of the ``keras.models.Model`` to export.

