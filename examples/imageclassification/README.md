# Image Classification Scenario

The Image Classification Tutorial maps the corresponding [Tensorflow Tutorial](https://www.tensorflow.org/tutorials/images/classification) to _tfaip_.
Improvements such as data augmentation and dropout are omitted in this tutorial and can be integrated as an exercise.

## Run
To run the training of this scenario execute (in the cloned dir)
```bash
export PYTHONPATH=$PWD  # required so that the scenario is detected

# Training
tfaip-train examples.imageclassification --trainer.output_dir ic_model
tfaip-train examples.imageclassification --trainer.output_dir ic_model --device.gpus 0  # to run training on the first GPU, if available
tfaip-train examples.imageclassification --model.conv_filters 30 50 60 --model.dense 200 200 --trainer.output_dir ic_model --device.gpus 0  # try a different (larger) model

# TensorBoard can be used to view the training process
tensorboard --logdir .
# Open http://localhost:6006

# Prediction
tfaip-predict --export_dir ic_model/best --data.image_files examples/imageclassification/examples/592px-Red_sunflower.jpg 
# Possible Output >>> This image most likely belongs to sunflowers with a 0.83 percent confidence.


```

## References
* Official [Tensorflow Tutorial](https://www.tensorflow.org/tutorials/images/classification)
* PLANET AI GmbH offers an [Intelligent Image Analysis](https://planet-ai.de/applications/image-analysis/) which is able to localize and identify many visual categories in images and videos.
  This skill is often utilized by our engines to preprocess images or videos to segment certain objects, but can also be used as a stand-alone solution.
