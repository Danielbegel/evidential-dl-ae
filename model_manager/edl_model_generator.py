from tensorflow.keras import Model
from tensorflow.keras import layers
import tensorflow as tf
from model_evaluation import loss_calculator


class EDLModel(Model):
    def __init__(self, train_config):
        super(VaeModel, self).__init__()

        #TODO 

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        mse = tf.keras.losses.MeanSquaredError()
        reconstruction_loss = mse(inputs, reconstructed) * self.input_shape[0]
        # Source : https://keras.io/examples/generative/vae/
        kl_loss = loss_calculator.get_vae_kl_loss(z_log_var, z_mean)
        total_loss = reconstruction_loss + kl_loss
        self.add_loss(total_loss)
        return reconstructed


class Sampling(layers.Layer):
    """Uses the  basic z_mean, z_log_var to sample"""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
