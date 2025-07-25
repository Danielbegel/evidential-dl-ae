from model_manager.ae_model_generator import AeModel
from model_manager.edl_model_generator import EDLModel
# from model_manager.vae_model_generator import VaeModel

def create_model(train_config, model_type):
    """
    This is a sample method that shows us how to use the various models created as a part of this code base,
    we can use any form for model creation, but it would be better if you stuck to subclassing which is a better practice
    """
    if model_type == "AE":
        return AeModel(train_config)
    elif model_type == "EDL":
        print("creating EDL model...")
        return EDLModel(train_config)
    # elif model_type == "VAE":
    #     return EDLModel(train_config)
    else:
        raise Exception("Invalid Model Name")

