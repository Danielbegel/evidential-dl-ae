import tensorflow as tf
import keras
import matplotlib.pyplot as plt

def get_reconstruction_losses(model, data):
    reconstructed = model(data)
    errors = tf.reduce_mean(tf.math.squared_difference(data, reconstructed), axis=1)
    return errors.numpy()