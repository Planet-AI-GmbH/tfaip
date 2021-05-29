# Full Tutorial

This tutorial shows how to use and implement optional by handy features of *tfaip*.
Go to the [minimal tutorial](../min) to see a scenario that only implements the required classes and functions.

This tutorial sets up training on the MNIST data which can then be used to predict digits of image files.

## Overview

The following features are covered by this tutorial

* setting up of a DataPipeline using DataProcessors, see [here](data).
* setting up of different data generation for [training](data/training_data_generation.py) and [prediction](data/prediction_data_generation.py), see also [here](data).
* selection and configuration of [different dynamic graphs](#dynamic-graphs)
* Writing image files to the tensorboard, see [here](model.py)
* setting up a Predictor that can vote the predictions of multiple individual models, see [here](predictor.py)
* Evaluator


## Dynamic Graphs

Dynamic graphs allow to change and setup layers with parameters that can be set in the command line.

* First, setup a static [Graph](graphs/tutorialgraph.py) which will handle the creation of the dynamic layers.
  For MNIST, this graph also adds the final output as additional layer since it is obligatory.
  Furthermore, the data is normalized and reshaped.
* Next, create a [base class and base params](graphs/backend.py) which is derived from `keras.layers.Layer`
  and must layer be implemented by each variant.
  Add an abstract method to the parameters to define how to create the layer.
  Here (`cls()`), only the class type is returned while assuming that the first and only argument of the `__init__` is the parameter.
  Optionally define a generic `TypeVar` for the parameters that can be used to define the actual parameter type in the actual implemented layer.
* Now, implement the base class and base parameters.
  In the tutorial, a [CNN](graphs/cnn.py) and [MLP](graphs/mlp.py) setup is provided.
* Finally, add a parameter to select the layers to the base params, here called `graph` in the `ModelParams`.
  Optionally, set the `choices` flag of `pai_meta` to provide the list of available parameters that can be selected.
  The static [Graph](graphs/tutorialgraph.py) calls the abstract `cls()` method to retrieve the actual implementation
  and instantiates it.
