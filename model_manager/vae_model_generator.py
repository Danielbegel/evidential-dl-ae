from tensorflow.keras import Model
from tensorflow.keras import layers
import tensorflow as tf
from keras import ops



class VAEModel(Model):
    #Replace AE's latent space with statistical distribution.
    # Source : https://keras.io/examples/generative/vae/



    def __init__(self, train_config):
        super(VAEModel, self).__init__()
        self.latent_dim = train_config["encoder_design"]["latent_layer_dimension"]
        self.input_shape = (train_config["encoder_design"]["input_layer_dimension"],)
        self.train_config = train_config
        self.kl_loss_metric = tf.keras.metrics.Mean("kl_loss", dtype=tf.float32) # initialize kl loss tracker as metric object




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



        # Combined Model
        self.encoder = Model(self.encoder_input, [self.z_mean, self.z_log_var, self.z])
        self.encoder.summary()
        self.decoder = Model(self.decoder_input, self.decoder_output)
        self.decoder.summary()





    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        mse = tf.keras.losses.MeanSquaredError()
        reconstruction_loss = mse(inputs, reconstructed)
        kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
        kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
        kl_annealing = self.train_config["hyperparameters"]["kl_annealing"]
        total_loss = reconstruction_loss + kl_loss*kl_annealing
        self.add_loss(total_loss)
        return reconstructed
    
    
    def get_kl_loss(model):
        return []


class Sampling(layers.Layer):
    """Uses the  basic z_mean, z_log_var to sample"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon