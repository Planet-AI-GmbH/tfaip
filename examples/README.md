# TFAIP Examples

This directory comprises some tutorials of few use-cases based on _tfaip_.
The use-cases provide only basic implementations which must be clearly extended for an actual real-world application.
The examples orientate at the [tutorials of Tensorflow](https://www.tensorflow.org/tutorials/).

## Requirements

The examples have additional requirements that can be installed by running:

```shell
pip install -re examples/requirements.txt
```

## List of Scenarios
* [MNIST:](tutorial) MNIST scenario designed as tutorial to show the [basic](tutorial/min) or [extended](tutorial/full) features of _tfaip_.
* [Template:](template) Templates to set up new scenarios
* [Image Classification:](imageclassification) Image classification of flowers of five classes (corresponding Tensorflow tutorial is available [here](https://www.tensorflow.org/tutorials/images/classification))
* [ATR:](atr) Line-based Automatic Text Recognition
* [Text/Fine-tuning BERT:](text/finetuningbert) Fine-tuning of BERT on the task that decides whether two sentences are semantically equivalent.
