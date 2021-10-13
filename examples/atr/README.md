# ATR Scenario

The ATR-Scenario is an example showing how to implement a line-based ATR-engine.
It provides a CNN/LSTM-network architecture which is trained with the CTC-algorithm.
This tutorial shows only the fundamentals and does not include required algorithms for document analysis in general
which are part of a real OCR/Document-Analysis engine.

## Run
To run the training of this scenario execute (in the cloned dir)
```bash
export PYTHONPATH=$PWD  # required so that the scenario is detected

# Training
tfaip-train examples.atr --trainer.output_dir atr_model
tfaip-train examples.atr --trainer.output_dir atr_model --device.gpus 0  # to run training on the first GPU, if available

# Validation (of the best model)
tfaip-lav --export_dir atr_model/best --data.image_files examples/atr/workingdir/uw3_50lines/test/*.png

# Prediction
tfaip-predict --export_dir atr_model/best --data.image_files examples/atr/workingdir/uw3_50lines/test/*.png
```

Note, the prediction will only print the raw output of the network.

## Data
The [working dir](workingdir) provides some example lines of the UW3 dataset which are loaded by default

## References
* PLANET AI GmbH offers an intelligent [Document Analysis Suite (IDA)](https://planet-ai.de/applications/document-analysis/) which is able to read and even understand a broad spectrum of documents from ancient hand-written documents to modern machine-generated ones.