#TODO: port eval and plot making code
import tensorflow as tf
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from keras import ops

'''This file produces plots'''


# def evaluate_model(train_config, model, data, model_type):
#     if model_type == 'AE':
#         results = model.evaluate(data, data)
#     else:
#         results = model.evaluate(data)
#     return results





def calculate_loss(train_config, model, data, model_type):

    data = data[:train_config["data_split"]["max_data"]] # truncate data



    if model_type == 'EDL': # calculate loss for EDL model
        one_hot_labels = np.column_stack((np.ones(len(data)),np.zeros(len(data))))
        data = (data, one_hot_labels)


        reconstructed_data = model(data).numpy()
        mse_loss = (reconstructed_data - data[0]) ** 2
        mse_loss = np.mean(mse_loss, axis=1, keepdims=True)



        z, alpha = model.encoder(data[0]) # get z and alpha vector from encoder
        alpha = tf.clip_by_value(alpha, 1e-4, 1e4) # clip alpha to avoid numerical instability near 0
        prior_alpha = tf.broadcast_to(model.prior_alpha, tf.shape(alpha)) # reshape prior alpha to match the dims of alpha
        prior_alpha = tf.clip_by_value(prior_alpha, 1e-4, 1e4) # clip prior_alpha to avoid numerical instability near 0
        

        '''KL loss'''
        truth_labels = data[1]
        tilde_alpha = truth_labels + (1 - truth_labels) * alpha
        tilde_alpha = tf.where(tf.math.is_finite(tilde_alpha), tilde_alpha, tf.ones_like(tilde_alpha) * 1.0)
        tilde_alpha = tf.clip_by_value(tilde_alpha, 1e-4, 1e6)
        kl_annealing = model.train_config["hyperparameters"]["kl_annealing"] # define kl scaling factor from train_config
        # calculate the statistical differenfce between alpha and prior alpha
        kl_loss = tf.reduce_sum(
                                            tfp.distributions.Dirichlet(tilde_alpha).kl_divergence(
                                            tfp.distributions.Dirichlet(prior_alpha))
                                        )
        kl_loss = (kl_loss * kl_annealing).numpy()
            
        return mse_loss, kl_loss
    
    elif model_type == 'VAE': # calculate loss for VAE model
        reconstructed_data = model(data).numpy()
        mse_loss = (reconstructed_data - data) ** 2
        mse_loss = np.mean(mse_loss, axis=1, keepdims=True)


        z_mean, z_log_var, z = model.encoder(data)
        kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
        kl_loss = ops.mean(ops.sum(kl_loss, axis=1)).numpy()
        kl_loss = kl_loss * train_config["hyperparameters"]["kl_annealing"]
        
        return mse_loss, kl_loss

    elif model_type == 'AE':
        reconstructed_data = model(data).numpy()
        mse_loss = (reconstructed_data - data) ** 2
        mse_loss = np.mean(mse_loss, axis=1, keepdims=True)
        return mse_loss
    
    else:
        raise NotImplemented



    
def calculate_probability_uncertainity(train_config, model, data):
    raise NotImplementedError #TODO implement