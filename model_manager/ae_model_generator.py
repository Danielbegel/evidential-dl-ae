from tensorflow.keras import Model
from tensorflow.keras import layers
import tensorflow as tf
from model_evaluation import loss_calculator


class AeModel(Model):
    def __init__(self, train_config):
        super(VaeModel, self).__init__()
        self.latent_dim = train_config["encoder_design"]["latent_layer_dimension"]
        self.input_shape = (train_config["encoder_design"]["input_layer_dimension"],)

        # Encoder
        self.encoder_input = layers.Input(shape=self.input_shape)
        x = self.encoder_input
        ip_layer_dimension_list = train_config["encoder_design"]["inner_layer_dimensions"]
        for dimension in ip_layer_dimension_list:
            x = layers.Dense(dimension, activation='relu', name=f"encoder_{dimension}")(x)
        self.z_mean = layers.Dense(self.latent_dim)(x)
        self.z_log_var = layers.Dense(self.latent_dim)(x)
        self.z = Sampling()([self.z_mean, self.z_log_var])

        # Decoder
        self.decoder_input = layers.Input(shape=(self.latent_dim,))
        op_layer_dimension_list = train_config["decoder_design"]["inner_layer_dimensions"]
        x = 0
        for dimension in op_layer_dimension_list:
            if x == 0:
                x = self.decoder_input
            x = layers.Dense(dimension, activation='relu')(x)
        self.decoder_output = layers.Dense(self.input_shape[0], activation='sigmoid')(x)

        # Combine
        self.encoder = Model(self.encoder_input, [self.z_mean, self.z_log_var, self.z])
        self.encoder.summary()
        self.decoder = Model(self.decoder_input, self.decoder_output)
        self.decoder.summary()

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
