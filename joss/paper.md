---
title: '_tfaip_ - a Generic and Powerful Research Framework for Deep Learning based on Tensorflow'
tags:
    - Python
    - Deep Learning
    - Tensorflow
    - Keras
    - Research
    - High-Level Framework
    - Generic
authors:
    - name: Christoph Wick
      orcid: 0000-0003-3958-6240
      affiliation: "1"
    - name: Benjamin Kühn
      affiliation: "1"
    - name: Gundram Leifert
      affiliation: "1"
    - name: Konrad Sperfeld
      affiliation: "2"
    - name: Tobias Strauß
      affiliation: "1"
    - name: Jochen Zöllner
      orcid: 0000-0002-3889-6629
      affiliation: "1,2"
    - name: Tobias Grüning
      orcid: 0000-0003-0031-4942
      affiliation: "1"
affiliations:
    - name: Planet AI GmbH, Warnowufer 60, 18059 Rostock, Germany
      index: 1
    - name: Institute of Mathematics, University of Rostock, 18051 Rostock, Germany
      index: 2
date: 29 March 2021
bibliography: paper.bib
---

# Summary

_tfaip_ is a Python-based research framework for developing, structuring, and deploying Deep Learning projects powered by Tensorflow [@tensorflow2015-whitepaper] and is intended for scientists of universities or organizations who research, develop, and optionally deploy Deep Learning models.
_tfaip_ enables to implement both simple and complex scenarios, such as image classification, object detection, text recognition, natural language processing, or speech recognition.
Each scenario is highly configurable by parameters that can directly be modified by the command line or the API.

# Statement of Need

An application of _tfaip_ resolves recurrent obstacles of research and development in an elegant and robust way:

* A complete scenario including the graph (e.g., the network architecture), the training (e.g., the optimizer or learning rate schedule), and the data pipeline (e.g., the data sources) are fully parameterizable by the command line or the API.
  This simplifies hyper-parameter-optimization but also testing of novel ideas.
* Each scenario is created by implementing predefined interfaces (e.g., loss-function or the graph construction).
  This leads to a clean, structured, modularized, and readable code preventing bad code style and facilitates maintenance.
* _tfaip_ provides a simple API to deploy a scenario.
  The corresponding module will automatically apply pre-processing, infer the trained model, and optionally transform the output by a post-processing pipeline. 
  The information about the pipeline-construction is embedded within the model which enables to store and load models with a different data pipeline even for the same scenario.
  This is handy if, for example, certain pre-processing steps are not required for one specific model or other inputs are expected.
* During research, a tedious step is data preparation which often comprises the conversion of data into the format required by the framework.
  Tensorflow allows integrating Python code in the data pipeline which is however not run (truly) in parallel by multiple processes and results quite often in a bottleneck.
  To speed-up Tensorflow, a user has to transform Python into Tensorflow operations which is laborious, partly even impossible, and complicates debugging.
  _tfaip_ tackles this issue by providing a sophisticated pipeline setup based on so-called data processors which apply simple transformation operations in pure Python code and are automatically executed in parallel.
  
# State of the Field

Efficient research in the area of Deep Learning requires the integration of highly sophisticated Open-Source frameworks such as [Tensorflow](https://www.tensorflow.org/) [@tensorflow2015-whitepaper], [PyTorch](https://pytorch.org/) [@pytorch_2019], [Caffe](https://github.com/BVLC/caffe) [@jia2014caffe], [CNTK](https://github.com/microsoft/CNTK) [@cntk2016], or [Trax](https://github.com/google/trax) [@trax2021].
These frameworks provide efficient tools to freely design Deep Learning scenario of any size and complexity.
However, as the number of code lines rises, a project has to be structured into meaningful components to be maintainable.
These components are almost identical for each Deep Learning scenario: there are modules for the graph, the model, the data, the training, the validation, and the prediction (i.e., the application of a trained model).
Furthermore, support for dynamic parameter adaption for instance via the command line is desirable for efficient research.
Therefore, to obtain a clean code base, it is highly desirable to only implement abstract templates that already set up the interaction among the modules by providing basic functionality that is required in any use-case.
_tfaip_ which is an extension to Tensorflow solves this and thus helps developers to efficiently handle and maintain small but also large-scale projects in research environments.

# _tfaip_ Functionality

The following itemization highlights the main functionality of _tfaip_:

* A basic concept of _tfaip_ is to split parameters and their actual object whereby the parameters are used to instantiate the corresponding object.
  This allows to build a hierarchical parameter tree where each node can be replaced with other parameters.
  Each parameter and also the replacements can be defined via the command line which enables to dynamically adapt, for example, even complete graphs of a model.
  In a research environment this simplifies hyper-parameter search and the setup of experiments.
* Class-templates define the logical structure of a real-world scenario:
    * A scenario requires the definition of a model and a data class.
      Basic functionality such as training, exporting, or loading of a model using the data is already provided.
    * To set up the data pipeline, a data generator and a list of data processors have to be implemented that define how raw data flows.
      The prepared data is then automatically fed into the neural network.
      Since each data processor is written in pure Python, it is simple to set up and debug the data processing pipeline.
      To speed up the data sample generation the pipeline can automatically be run in parallel.
    * The model comprises information about the loss, metrics, and the graph of the scenario.
      The model template hereby requires the user to implement methods, the superordinate modules are then connected automatically.
* _tfaip_ tracks the training and validation process by utilizing the [Tensorboard](https://www.tensorflow.org/tensorboard/) provided by Tensorflow.
  The Tensorboard can be extended by custom data, for example by plotting Precision-Recall (PR) curves or rendering images.
* The prediction module loads a trained scenario so that it can easily be applied on new data during deployment.
  Since a model stores its pre- and post-processing pipeline no additional data handling has to be performed.
* Because each scenario follows a predefined setup, shared research code is clearer and consequently can be easier reviewed, extended, or applied.
  For example, this modularity simplifies the process if several users are working together on the same scenario.
* An important feature of _tfaip_ is the consistent use of Python's typing module including type checks.
  This leads to clean understandable code and fewer errors during development.
  Furthermore, this enables IDEs such as [PyCharm](https://www.jetbrains.com/pycharm/) to perform autocompletion.
* In some rare cases, the highly generic API of _tfaip_ might not be sufficient.
  To tackle this, each scenario can optionally customize any functionality by implementing the base classes, for example the trainer or the data pipeline.

# _tfaip_ Documentation and Tutorials

To help new users to become familiar with _tfaip_, a [comprehensive documentation](https://tfaip.readthedocs.io/), several [tutorials](https://github.com/Planet-AI-GmbH/tfaip/tree/master/examples/tutorial), and [example scenarios](https://github.com/Planet-AI-GmbH/tfaip_example_scenarios) with real-world use-cases are available:

* _tfaip_ provides two tutorials that solve the classification of MNIST: 
  the _minimal_ scenario shows the minimal implementation that is required to implement the common MNIST-tutorial in _tfaip_.
  The _full_ scenario implements the same use-case and highlights different advanced features of _tfaip_.
* Some more [examples](https://github.com/Planet-AI-GmbH/tfaip_example_scenarios) are provided by transferring official [Tensorflow tutorials](https://www.tensorflow.org/tutorials) in the _tfaip_ framework.
  These scenarios show the power of the framework as complex scenarios are logically split into meaningful parts and components.
  The examples comprise:
  * Automatic Text Recognition (ATR) of single text line images
  * Image Classification
  * Fine Tuning of a BERT 
* [Templates](https://github.com/Planet-AI-GmbH/tfaip/tree/master/examples/template) for two basic scenarios allow setting up a new scenario by copying basic code and modifying it afterwards. 
  All required files and classes are already set up, solely the abstract methods have to be implemented and classes should be renamed.

# Usage of _tfaip_ in Research

Diverse active research projects are already based on _tfaip_:

* The [NEISS-project](https://github.com/NEISSproject/tf2_neiss_nlp) integrated _tfaip_ to solve their Natural Language Processing (NLP) problems.
* Since its 2.0 release, the open-source ATR engine [Calamari](https://github.com/calamari_ocr/calamari) by @wick_calamari_2020 is based on _tfaip_.
* Our research, for example a recent publication on ATR using Transformers, uses _tfaip_ [@wick_bidirectional_2021].


# Acknowledgments

The authors would like to thank the open-source community, especially the developers and maintainers of Python, Tensorflow, and Numpy, since these packages empower _tfaip_.

This work was partially funded by the European Social Fund (ESF) and the Ministry of Education, Science and Culture of Mecklenburg-Western Pomerania (Germany) within the project NEISS under grant no ESF/14-BM-A55-0006/19.

# References
