import os


def generate_model_name(train_config):
    """
    This function generates the model name based on the criteria below
    """
    model_prefix = "benchmark_{}{}_e{}.keras"

    layer_values = ""
    inner_layer_dimensions = train_config["encoder_design"]["inner_layer_dimensions"]
    print(len(inner_layer_dimensions))
    if len(inner_layer_dimensions) != 0:
        model_prefix = "benchmark_{}_{}_e{}.keras"
        layer_values = '_'.join([str(val) for val in inner_layer_dimensions])
    model_name = model_prefix.format(train_config["encoder_design"]["latent_layer_dimension"], layer_values,
                                     train_config["hyperparameters"]["epochs"])
    return model_name


def save_model(train_config, trained_model):
    """
    This function saves the model based on the predefined format, we can choose from various methods for saving models
    """
    model_name = train_config["model_files"]["model_name"]
    print("Saving Model {}".format(model_name))

    if model_name == "":
        model_name = generate_model_name(train_config)
    trained_model.save(os.path.join(train_config["model_files"]["model_location"], model_name + ".keras"))
