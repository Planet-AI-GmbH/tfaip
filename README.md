[![Python Test](https://github.com/Planet-AI-GmbH/tfaip/actions/workflows/python-test.yml/badge.svg)](https://github.com/Planet-AI-GmbH/tfaip/actions/workflows/python-test.yml)
[![Python Test](https://github.com/Planet-AI-GmbH/tfaip/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Planet-AI-GmbH/tfaip/actions/workflows/python-publish.yml)

# _tfaip_ - A Generic and Powerful Research Framework for Deep Learning based on Tensorflow

*tfaip* is a Python-based research framework for developing, organizing, and deploying Deep Learning models powered by [Tensorflow](https://www.tensorflow.org/).
It enables to implement both simple and complex scenarios that are structured and highly configurable by parameters that can directly be modified by the command line (read the [docs](https://tfaip.readthedocs.io)).
For example, the [tutorial.full](examples/tutorial/full)-scenario for learning MNIST allows to modify the graph during training but also other hyper-parameters such as the optimizer:
```bash
export PYTHONPATH=$PWD  # set the PYTHONPATH so that the examples dir is found
# Change the graph
tfaip-train examples.tutorial.full --model.graph MLP --model.graph.nodes 200 100 50 --model.graph.activation relu
tfaip-train examples.tutorial.full --model.graph MLP --model.graph.nodes 200 100 50 --model.graph.activation tanh
tfaip-train examples.tutorial.full --model.graph CNN --model.graph.filters 40 20 --model.graph.dense 100
# Change the optimizer
tfaip-train examples.tutorial.full --trainer.optimizer RMSprop --trainer.optimizer.beta1 0.01 --trainer.optimizer.clip_global_norm 1
# ...
```

A trained model can then easily be integrated in a workflow to predict provided `data`:
```python
predictor = TutorialScenario.create_predictor("PATH_TO_TRAINED_MODEL", PredictorParams())
for sample in predictor.predict(data):
    print(sample.outputs)
```

In practice, _tfaip_ follows the rules of object orientation, i.e., the code for a scenario (e.g., image-classification (MNIST), text recognition, NLP, etc.) is organized by implementing classes.
By default, each [`Scenario`](https://tfaip.readthedocs.io/en/latest/doc.scenario.html) must implement [`Model`](https://tfaip.readthedocs.io/en/latest/doc.model.html), and [`Data`](https://tfaip.readthedocs.io/en/latest/doc.data.html).
See [here](examples/tutorial/full) for the complete code to run the upper example for MNIST and see [here](examples/tutorial/min) for the minimal setup.


## Setup

To setup _tfaip_ create a virtual Python (at least 3.7) environment and install the `tfaip` pip package: `pip install tfaip`:
```bash
virtualenv -p python3 venv
source venv/bin/activate
pip install tfaip
```
Have a look at the [wiki](https://tfaip.readthedocs.io/en/latest/doc.installation.html) for further setup instructions.

## Run the Tutorial

After the setup succeeded, launch a training of the tutorial which is an implementation of the common MNIST scenario:
```bash
export PYTHONPATH=$PWD  # set the PYTHONPATH so that the examples dir is found
tfaip-train examples.tutorial.full
# If you have a GPU, select it by specifying its ID
tfaip-train examples.tutorial.full --device.gpus 0
```

## Next Steps

Start reading the [Minimum Tutorial](examples/tutorial/min), optionally have a look at the [Full Tutorial](examples/tutorial/full) to see more features.
The [docs](https://tfaip.readthedocs.io/en/latest) provides a full description of `tfaip`.

To set up a _new custom scenario_, copy the [general template](examples/template/general) and implement the abstract methods.
Consider renaming the classes!
Launch the training by providing the path or package-name of the new scenario which _must_ be located in the `PYTHONPATH`!

## Features of _tfaip_

_tfaip_ provides different features which allow designing generic scenarios with maximum flexibility and high performance.

### Code design

* _Fully Object-Oriented_: Implement classes and abstract functions or overwrite any function to extend, adapt, or modify its default functionality.
* _Typing support_: _tfaip_ is fully typed with simplifies working with an IDE (e.g., use PyCharm!).
* Using pythons `dataclasses` module to set up parameters which are automatically converted to parameters of the command line by our [`paiargparse`](https://github.com/Planet-AI-GmbH/paiargparse) package.

### Data-Pipeline
Every scenario requires the setup of a data-pipeline to read and transform data.
*tfaip* offers to easily implement and modify even complex pipelines by defining multiple `DataProcessors` which usually implement a small operation to map an input sample to an output sample.
E.g., one `DataProcessor` loads the data (`input=filename`, `output=image`), another one applies normalization rules, again another one applies data augmentation, etc.
The **great advantage** of this setup is that the data processors run in Python and can automatically be parallelized by *tfaip* for speed up by setting `run_parallel=True`.

### Deep-Learning-Features

Since _tfaip_ is based on Tensorflow the full API are available for designing models, graphs, and even data pipelines.
Furthermore, *tfaip* supports additional common techniques for improving the performance of a Deep-Learning model out of the box:

* Warm-starting (i.e., loading a pretrained model)
* EMA-weights
* Early-Stopping
* Weight-Decay
* various optimizers and learning-rate schedules

## Contributing

We highly encourage users to contribute own scenarios and improvements of _tfaip_.
Please read the [contribution guidelines](https://tfaip.readthedocs.io/en/latest/doc.development.html).

## Benchmarks

All timings were obtained on a Intel Core i7, 10th Gen CPU.

### MNIST

The following Table compares the MNIST Tutorial of Keras to the [Minimum Tutorial](examples/tutorial/min).
The keras code was adopted to use the same network architecture and hyperparemter settings (batch size of 16, 10 epochs of training).

Code | Time Per Epoch | Train Acc | Val Acc | Best Val Acc
:---- | --------------: | ---------: | -------: | ------------: 
Keras |  16 s | 99.65% | 98.24% | 98.60% 
_tfaip_ | 18 s |  99.76% | 98.66% | 98.66% 

_tfaip_ and Keras result in comparable accuracies, as to be expected since the actual code for training the graph is fundamentally identical.
_tfaip_ is however a bit slower due some overhead in the input pipeline and additional functionality (e.g., benchmarks, or automatic tracking of the best model).
This overhead is negligible for almost any real-world scenario because due to a clearly larger network architecture, the computation times for inference and backpropagation become the bottleneck. 

### Data Pipeline

Integrating pure-python operations (e.g., numpy) into a `tf.data.Dataset `to apply high-level preprocessing is slow by default since [tf.data.Dataset.map](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map) in cooperation with [tf.py_function](https://www.tensorflow.org/api_docs/python/tf/py_function) does not run in parallel and is therefore blocked by Python's GIL.
_tfaip_ curcumvents this issue by providing an (optional) parallelizable input pipeline.
The following table shows the time in seconds for two different tasks:

* PYTHON: applying some pure python functions on the data
* NUMPY: applying several numpy operations on the data


|         Mode        |     Task     |     Threads 1      |     Threads 2      |     Threads 4      |     Threads 6      |
|:---------------------|:--------------|--------------------:|--------------------:|--------------------:|--------------------:|
| tf.py_function |    PYTHON    | 23.47| 22.78 | 24.38  | 25.76  |
|     _tfaip_    |    PYTHON    | 26.68| 14.48 |  8.11  | 8.13  |
| tf.py_function |    NUMPY     | 104.10 | 82.78  | 76.33  | 77.56  |
|     _tfaip_    |    NUMPY     | 97.07  | 56.93  | 43.78 | 42.73  |

The PYTHON task clearly shows that `tf.data.Dataset.map` is not able to utilize multiple threads.
The speed-up in the NUMPY tasks occurs possibly due to paralization in the numpy API to C.

