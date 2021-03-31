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
