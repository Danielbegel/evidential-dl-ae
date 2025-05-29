# Evidential DL in AEs for Anomaly Detection

## Overview
ae_model_entrypoint.py, vae_model_entrypoint.py and edl_model_entrypoint.py are example entrypoint files  for training models in this project. The script is designed to be a template, which can be extended to support multiple models. 


## Features

### `config`
Contains .json files that include model dimension, number of layers, hyperparameters, number of training epochs, etc. 

### `model_manager`
Contains model infrastructure to create new AE, VAE, or EDL models, based on training .json.

### `plotter`
Make standard plots.

## Getting Started
Prerequisites
Python 3.5+, h5py, tensorflow 

## Installation
```
git clone https://gitlab.cern.ch/schynowe/model_traineval_framework

cd model_traineval_framework
```

## Contributing
- Create a new branch and add changes
- Submit a pull request / or merge if you own the repo
