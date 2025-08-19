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
        self.LATENT_DIM = 2 # This is fixed since we have just two classes
        self.train_config = train_config
        encoder_dimensions = train_config["encoder_design"]["inner_layer_dimensions"]
        decoder_dimensions = train_config["decoder_design"]["inner_layer_dimensions"]
        self.kl_loss_metric = tf.keras.metrics.Mean("kl_loss", dtype=tf.float32) # initialize kl loss tracker as metric object
        self.kl_strength = 0.1 # starting value for KL strength


        # Encoder
        self.encoder_input = layers.Input(shape=self.input_shape)
        x = self.encoder_input
        for dim in encoder_dimensions:
            self.encoder_x = layers.Dense(dim, activation='relu')(x)
            x = self.encoder_x
        self.alpha = layers.Dense(self.LATENT_DIM, activation='softplus')(self.encoder_x)
        self.z = Sampling()(self.alpha)





        # Decoder
        self.decoder_input = layers.Input(shape=(self.LATENT_DIM,))
        x = self.decoder_input
        for dim in decoder_dimensions:
            self.decoder_x = layers.Dense(dim, activation='relu')(x)
            x = self.decoder_x
        self.decoder_output = layers.Dense(self.input_shape[0], activation='relu')(self.decoder_x)





        # Define enconder, decoder and prior_alpha
        self.encoder = Model(self.encoder_input, [self.z, self.alpha])
        self.decoder = Model(self.decoder_input, self.decoder_output)
        self.prior_alpha = tf.constant([1.0] * self.LATENT_DIM, dtype = tf.float32)




    def call(self, inputs):
        inputs, truth_labels = inputs # unpack the data, since it is passed in with truth values attached in the form of a tuple
        


        '''Get alpha'''
        z, alpha = self.encoder(inputs) # get z and alpha vector from encoder
        alpha = tf.where(tf.math.is_finite(alpha), alpha, tf.ones_like(alpha) * 1.0)
        alpha = tf.clip_by_value(alpha, 1e-6, 1e6) # clip alpha to avoid numerical instability near 0
        prior_alpha = tf.broadcast_to(self.prior_alpha, tf.shape(alpha)) # reshape prior alpha to match the dims of alpha
        
        
        # tf.print("truth: ", truth_labels)
        # tf.print("alpha: ", alpha)

        '''KL loss'''
        # calculate the statistical differenfce between alpha and prior alpha
        tilde_alpha =  truth_labels + (1 - truth_labels) * alpha
        tilde_alpha = tf.where(tf.math.is_finite(tilde_alpha), tilde_alpha, tf.ones_like(tilde_alpha) * 1.0)
        tilde_alpha = tf.clip_by_value(tilde_alpha, 1e-4, 1e6)



        kl_loss = tf.reduce_sum(
                                tfp.distributions.Dirichlet(alpha).kl_divergence(
                                tfp.distributions.Dirichlet(prior_alpha))
        )

        # tf.print("tilde_alpha: ", tilde_alpha)
        # tf.print("kl loss:", kl_loss)



        '''Scaling'''
        kl_annealing = self.train_config["hyperparameters"]["kl_annealing"] # define kl scaling factor from train_config
        kl_strength = self.kl_strength # updated after every epoch


        # kl_loss = kl_loss * kl_strength * kl_annealing # scale KL-loss accordingly TODO debug
        kl_loss = kl_loss *kl_strength + kl_annealing # scale KL-loss accordingly TODO debug



        self.kl_loss_metric.update_state(kl_loss) # update kl_loss tracker



        '''MSE loss'''
        reconstructed = self.decoder(z)  # get reconstructed inputs
        reconstructed = tf.where(tf.math.is_finite(reconstructed),
                         reconstructed,
                         tf.zeros_like(reconstructed))
        reconstructed = tf.clip_by_value(reconstructed, -1e3, 1e3)
        mse = tf.keras.losses.MeanSquaredError() # define MSE loss
        reconstruction_loss = mse(inputs, reconstructed) # calculate MSE loss



        '''Total loss'''
        total_loss = reconstruction_loss + kl_loss # add MSE and KL losses to get total loss
        self.add_loss(total_loss) # add loss to total loss used for training



        ''' debugging... '''
        # These lines make sure that non of the values being passed into the model turn into NaNs
        tf.debugging.check_numerics(reconstruction_loss, "recon loss NaN")       
        tf.debugging.check_numerics(kl_loss, "KL loss NaN")
        tf.debugging.check_numerics(total_loss, "total_loss NaN") 

        return reconstructed # return reconstructed inputs

    def get_kl_loss(model):
        return []

    

class Sampling(layers.Layer):
    '''Uses the alpha dirichlet to sample'''

    def call(self, inputs):
        inputs = tf.clip_by_value(inputs,1e-4,1e6)
        return tfp.distributions.Dirichlet(inputs).sample()



