from tensorflow.keras import Model
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_probability as tfp


class EDLModel(Model):
    """
    This model replaces the VAE parameters with the Dirichlet Paramaters

    """
    def __init__(self, train_config):
        super(EDLModel, self).__init__()
        self.input_shape = (train_config["encoder_design"]["input_layer_dimension"],)
        self.latent_dim = 2 # This is fixed since we have just two classes
        self.train_config = train_config

        # Encoder
        self.encoder_input = layers.Input(shape=self.input_shape)
        dimensions = train_config["encoder_design"]["inner_layer_dimensions"]
        x = self.encoder_input
        for dim in dimensions:
            self.encoder_x = layers.Dense(dim, activation='relu')(x)
            x = self.encoder_x
        self.alpha = layers.Dense(self.latent_dim, activation='softplus')(self.encoder_x)
        self.z = Sampling()(self.alpha)

        # Decoder
        self.decoder_input = layers.Input(shape=(self.latent_dim,))
        op_layer_dimension_list = train_config["decoder_design"]["inner_layer_dimensions"]
        x = self.decoder_input
        for dim in op_layer_dimension_list:
            self.decoder_x = layers.Dense(dim, activation='relu')(x)
            x = self.decoder_x
        self.decoder_output = layers.Dense(self.input_shape[0], activation='relu')(self.decoder_x)

        self.encoder = Model(self.encoder_input, [self.z, self.alpha])
        self.decoder = Model(self.decoder_input, self.decoder_output)

    def call(self, inputs):
        alpha, z = self.encoder(inputs)
        kl_loss = tf.reduce_sum(tfp.distributions.Dirichlet(alpha).kl_divergence(tfp.distributions.Dirichlet(prior_alpha))) #TODO debug 
        strength = self.train_config["hyperparameters"]["kl_strength"]
        reconstructed = self.decoder(z)
        mse = tf.keras.losses.MeanSquaredError()
        reconstruction_loss = mse(inputs, reconstructed) * self.input_shape[0]

        total_loss = reconstruction_loss + strength * kl_loss
        self.add_loss(total_loss)
        return reconstructed


class Sampling(layers.Layer):
    """Uses the alpha dirichlet to sample"""

    def call(self, inputs):
        return tfp.distributions.Dirichlet(inputs).sample()
