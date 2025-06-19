# Evidential DL in AEs for Anomaly Detection

## Overview
ae_model_entrypoint.py and edl_model_entrypoint.py are example entrypoint files  for training models in this project. The script is designed to be a template, which can be extended to support multiple models. 


## Features

### `config`
Contains .json files that include model dimension, number of layers, hyperparameters, number of training epochs, etc. 

### `model_manager`
Contains model infrastructure to create new AE, VAE, or EDL models, based on training .json. Also includes content to evaluate models after training.

### `plotter`
Make standard plots.

## Getting Started
Requirements (make a conda env): python 3.5+, h5py, tensorflow, tensorflow_probability, tf_keras

1. Run the `train_entrypoint.py` script, modifying the script to specify your train config file ("TRAINER_CONFIG_FILE") and desired model type ("Create Model")
2. 

## Contributing
- Create a new branch and add changes
- Submit a merge request if your changes should be preserved for other users
