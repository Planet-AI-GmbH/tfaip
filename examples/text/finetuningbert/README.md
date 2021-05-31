# Fine Tuning BERT Tutorial

The Fine Tuning BERT Tutorial maps the corresponding [Tensorflow Tutorial](https://www.tensorflow.org/official_models/fine_tuning_bert) to _tfaip_.
Only the training of the BERT is covered in this Tutorial.
Application, i.e. prediction is missing.
Note that instead of the large BERT, this Tutorial used Albert from [Hugging Face](https://huggingface.co/albert-base-v2) as pretrained model.
The task is to decide whether to sentences are semantically equivalent., e.g. the two sentences

> The identical rovers will act as robotic geologists , searching for evidence of past water .

and

> The rovers act as robotic geologists , moving on six wheels .

are not equivalent. 


## Run
To run the training of this scenario execute (in the cloned dir)
```bash
export PYTHONPATH=$PWD  # required so that the scenario is detected

# Training
tfaip-train examples.text.finetuningbert --trainer.output_dir ftbert_model

# TensorBoard can be used to view the training process
tensorboard --logdir .
# Open http://localhost:6006
tfaip-monitor/pythonProject
```

## References
* Official [Tensorflow Tutorial](https://www.tensorflow.org/official_models/fine_tuning_bert)
* A model zoo of diverse pretrained networks is available at [Hugging Face](https://huggingface.co)
