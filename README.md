# Evidential DL in AEs for Anomaly Detection

## Overview
ae_model_entrypoint.py and edl_model_entrypoint.py are example entrypoint files  for training models in this project. The script is designed to be a template, which can be extended to support multiple models. References to the EDL learning can be found at https://github.com/FAIR4HEP/PFIN4UQAD (corresponding paper https://iopscience.iop.org/article/10.1088/2632-2153/ade51b)


## Features

### `config`
Contains .json files that include model dimension, number of layers, hyperparameters, number of training epochs, etc. 

### `data_manager`
Contains scripts that will fetch data from h5 files specified in config and split background data into training, test, and validation groups. Also contains scripts for splitting the background data into training, testing, and validation sets.

### `model_manager`
Contains model infrastructure to create new AE, VAE, or EDL models, based on training .json. Also includes content to evaluate models after training.

### `grapher`
Make standard plots.
- ROC Curve
- Loss histogram (can split KL and MSE as well)
- Loss vs Epoch during training

## Getting Started
0. Requirements (make a conda env): python 3.5+, h5py, tensorflow, tensorflow_probability, tf_keras
1. Download the datasets from Zenodo and copy to `data/` directory.
    a. background: https://zenodo.org/records/5046389
    b. A -> 4 lepton: https://zenodo.org/records/5046446
    c. h -> tau tau: https://zenodo.org/records/5061633
    d. h+ -> tau nu: https://zenodo.org/records/5061688
    f. LQ -> b tau: https://zenodo.org/records/5055454
2. Run the `train_entrypoint.py` script, modifying the script to specify your train config file ("TRAINER_CONFIG_FILE") and desired model type ("Create Model")


## Contributing
- Create a new branch and add changes
- Submit a merge request if your changes should be preserved for other users
