#TODO: port eval and plot making code
import tensorflow as tf
import keras
from keras import layers


def evaluate_model(train_config, model, data, model_type):
    if model_type == 'AE':
        results = model.evaluate(data, data)
    else:
        results = model.evaluate(data)
    return results