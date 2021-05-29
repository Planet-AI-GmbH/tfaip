# Minimal Tutorial

Welcome to the minimal tutorial which shows by the example of MNIST how _tfaip_ is structured and what a custom scenario must implement.

## Scenario

The central class is the `TutorialScenario` which is (optionally) parametrized by `TutorialScenarioParams`. Here no additional parameters are required for the overall scenario.
The `TutorialScenario` glues together the `Data`-Pipeline, the training data origin, and the `Model` with is `Graph`.

## Data

The [`TutorialData`](data.py) defines the input and target shapes of the `Scenario`.
In the case of MNIST, the input is an image labelled `img` with a shape of `28x28` and a `dtype` of `uint8`.
The targets (`gt`) are a simple scalar (use `shape=[1]`) with a `dtype` of `uint8`.

```python
class TutorialData(DataBase[DataBaseParams]):
    def _input_layer_specs(self):
        return {'img': tf.TensorSpec(shape=(28, 28), dtype='uint8')}

    def _target_layer_specs(self):
        return {'gt': tf.TensorSpec(shape=[1], dtype='uint8')}
```

## Training Data-Generation

The next step is to load the training data which is done in the `TutorialDataGenerator` (note that the corresponding parameter class will actually instantiate the Generator).
This first loads the `keras.dataset` and selects the desired partition depending on the pipeline mode.
A general `DataGenerator` must overwrite `__len__` to return the number of samples in the dataset and the `generate` to provide the samples.
Here, `generate()` just returns the loaded (and converted) samples.

```python
class TutorialDataGenerator(DataGenerator[TutorialDataGeneratorParams]):
    def __init__(self, mode: PipelineMode, params: 'TutorialDataGeneratorParams'):
        super().__init__(mode, params)
        dataset = getattr(keras.datasets, params.dataset)
        train, test = dataset.load_data()
        data = train if mode == PipelineMode.TRAINING else test
        self.data = to_samples(data)

    def __len__(self):
        return len(self.data)

    def generate(self) -> Iterable[Sample]:
        return self.data
```

Afterwards, parameters that define the `TrainerPipelineParamsBase` must be defined.
This parameter set defines how to create the training data and optionally the validation data.
Here, only one (`train_val`) `TutorialDataGeneratorParams`-set is used for training and validation.
Split this into two, if there are different parameters for training and validation.
```python
@pai_dataclass
@dataclass
class TutorialTrainerGeneratorParams(TrainerPipelineParamsBase[TutorialDataGeneratorParams, TutorialDataGeneratorParams]):
    train_val: TutorialDataGeneratorParams = field(default_factory=TutorialDataGeneratorParams, metadata=pai_meta(mode='flat'))

    def train_gen(self) -> TutorialDataGeneratorParams:
        return self.train_val

    def val_gen(self) -> Optional[TutorialDataGeneratorParams]:
        return self.train_val
```

Now, the training pipeline is set-up and the model and its graph can be defined.

## Model and Graph

The `TutorialModel` and its `TutorialGraph` are described by a corresponding `TutorialModelParams` class:
```python
@pai_dataclass
@dataclass
class TutorialModelParams(ModelBaseParams):
    n_classes: int = field(default=10, metadata=pai_meta(
        help="The number of classes (depends on the selected dataset)"))
```

Here, only the number of classes must be specified (which could be derived from data).

The `TutorialGraph` inherits `GraphBase` and uses the `TutorialModelParams`-structure.
A `GraphBase` is a `keras.Layer` so, as recommended by Tensorflow, first create the layers in the `__init__` function,
then connect them in the `call` method.
Here, a CNN with two conv and pool layers and two dense layers (FF) is created.
The `input` of the layer are of the shape that are defined in [`Data`](#data) ([above](#data)), therefore a dict with one entry of `'img'`.
The `call` function must also return a `dict` of tensors, its keys are later used to access the outputs (e.g. in the loss or the metric).

```python
class TutorialGraph(GraphBase[TutorialModelParams]):
    def __init__(self, params: TutorialModelParams, name='conv', **kwargs):
        super(TutorialGraph, self).__init__(params, name=name, **kwargs)
        self.conv1 = Conv2D(kernel_size=(2, 2), filters=16, strides=(1, 1), padding='same', name='conv1')
        self.pool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='pool1')
        self.conv2 = Conv2D(kernel_size=(2, 2), filters=32, strides=(1, 1), padding='same', name='conv2')
        self.pool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='pool2')
        self.flatten = keras.layers.Flatten()
        self.ff = FF(out_dimension=128, name='f_ff', activation='relu')
        self.logits = FF(out_dimension=self._params.n_classes, activation=None, name='classify')

    def call(self, inputs, **kwargs):
        rescaled_img = K.expand_dims(K.cast(inputs['img'], dtype='float32') / 255, -1)
        conv_out = self.pool2(self.conv2(self.pool1(self.conv1(rescaled_img))))
        logits = self.logits(self.ff(self.flatten(conv_out)))
        pred = K.softmax(logits, axis=-1)
        cls = K.argmax(pred, axis=-1)
        return {'pred': pred, 'logits': logits, 'class': cls}
```

The corresponding `TutorialModel` must define how to create the graph, here by just constructing a `TutorialGraph`.
Next, define the `_loss`: the `inputs_targets` are a joined dict of both the inputs and targets coming from the [`TutorialData`](#data).
The `outputs` are the output-dict of the previously defined `TutorialGraph`.
The loss is again an "output" of the graph and must therefore be wrapped in a `keras.Layer`, here, a `sparse_categorical_crossentropy` is wrapped within a `Lambda`-layer.
The metric is computed by selecting the `gt` (targets) and `class` (outputs) and passing them to a `keras.metrics.Accuracy()`.
There are different, also more complex methods to define metrics, see the [full tutorial](../full) for an example.
Both the loss and metric must return a dict. Its keys will be used for displaying information in the log or the TensorBoard.

Finally, optionally define how to determine the best model (`_best_logging_settings_`), here by selecting the model with the `"max"` `"acc"`, and
how to print useful human-readable information during validation.
The `_print_evaluate`-function receives a single sample (unbatched) and displays (here) the target, prediction and if the prediction was correct.

```python
class TutorialModel(ModelBase[TutorialModelParams]):
    def create_graph(self, params: TutorialModelParams) -> 'GraphBase':
        return TutorialGraph(params)

    def _loss(self, inputs_targets, outputs) -> Dict[str, AnyTensor]:
        return {'loss': tf.keras.layers.Lambda(
            lambda x: tf.keras.metrics.sparse_categorical_crossentropy(*x, from_logits=True), name='loss')(
            (inputs_targets['gt'], outputs['logits']))}

    def _metric(self):
        return {'acc': MetricDefinition("gt", "class", keras.metrics.Accuracy())}

    def _best_logging_settings(self):
        return "max", "acc"

    def _print_evaluate(self, inputs, outputs: Dict[str, AnyNumpy], targets: Dict[str, AnyNumpy], data, print_fn=print):
        correct = outputs['class'] == targets['gt']
        print_fn(f"PRED/GT: {outputs['class']}{'==' if correct else '!='}{targets['gt']} (p = {outputs['pred'][outputs['class']]})")
```

## Launch the Training
Training can be started by calling
```bash
tfaip-train tutorial.min
```

## Further reading

After having finished this Tutorial, have a look at the [full tutorial](../full) or the [wiki](../../../../wiki).
