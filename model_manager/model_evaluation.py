#TODO: port eval and plot making code
import tensorflow as tf
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp



def evaluate_model(train_config, model, data, model_type):
    if model_type == 'AE':
        results = model.evaluate(data, data)
    else:
        results = model.evaluate(data)
    return results

def calculate_loss(train_config, model, data, model_type):

    #truncate as desired
    data = data[:train_config["data_split"]["max_data"]]

    #Calculating MSE loss
    reconstructed_data = model(data).numpy()
    mse = (reconstructed_data - data) ** 2
    mse_avg = np.mean(mse, axis=1, keepdims=True)


    #Calculate KL loss if applicable
    if model_type == 'EDL':
        strength = train_config["hyperparameters"]["kl_strength"]
        num_classes = model.latent_dim
        prior_alpha_constant = tf.constant([1.0] * num_classes, dtype=tf.float32)

        alpha, _ = model.encoder(data)
        prior_alpha_broadcasted = tf.broadcast_to(prior_alpha_constant, tf.shape(alpha))
        kl_divergences_per_sample = tfp.distributions.Dirichlet(alpha).kl_divergence(
                                tfp.distributions.Dirichlet(prior_alpha_broadcasted))
        kl_divergence = (strength * kl_divergences_per_sample).numpy()

        return mse_avg, kl_divergence
    
    elif model_type == 'VAE':
        #TODO
        print("Not yet implemented...")
    else:
        return mse_avg
