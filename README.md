# tf2_aip_base
This repository is designed as a research framework for supervised machine learning. 
It aims to reduce your work on the train-loop, validation, saving, optimizer, multi-gpu and 
provides lot more features, which can be configured via command line.

# Setup
see: [Setup](https://github.com/Planet-AI-GmbH/tf2_aip_base/wiki/Install)
# Usage
To setup your own scenario see: [Scenario setup](https://github.com/Planet-AI-GmbH/tf2_aip_base/wiki/Scenario-setup.md)

## Running the tutorial scenario:
The default tutorial scenario is 'fashion-mnist'. Run your first training with:

`tfaip-train tutorial --trainer_params checkpoint_dir=models/fashion_default`

tfaip-train refers to `tfaip/scripts/train.py`. You can switch to mnist data with `--data_params dataset=mnist`.

You can evaluate the model on the validation dataset with:
`tfaip-lav --export_dir models/fashion_default/export`

Most hyper parameter can be configured via command line see `tfaip-train -h` and `tfaip-train tutorial -h`.
Checkout the [Wiki](https://github.com/Planet-AI-GmbH/tf2_aip_base/wiki) for further explanations.

_Contributions are welcome, and they are greatly appreciated!_




