# This is SAMPLE code that shows how this code can be used, DO NOT MAKE THIS THE TOGO CODE FOR PERMANENT CHNAGES!

# Imports
import json
from data_manager import fixed_splitter as fs
from config.data_manager import data_preprocessor as preproc
from model_manager import model_trainer, model_generator
from model_evaluation import model_evaluator
from config.data_manager import data_preprocessor
from model_evaluation import loss_calculator
from grapher import latent_space_graph_generator
import numpy as np
from sklearn.metrics import auc

print("Initializing Configuration for Training...")
TRAINER_CONFIG_FILE = 'config/trainer_config_v1.0.json'
model_type = "EDL"

# Initialize the Config here
with open(TRAINER_CONFIG_FILE, 'r') as file:
    train_config = json.load(file)

print("Configuration Initialized")

# Get Data
print("Preprocessing Data")
preprocessed_data = preproc.get_data(train_config)

# Create Splits
print("Splitting Data")
data_train, data_validate, data_test = fs.generate_fixed_trainingdata_split(train_config, preprocessed_data, True)

# Create Model
print("Creating Model")
edl_model = model_generator.create_model(train_config, model_type)

# Train Model
print("Training Model")
trained_model, history = model_trainer.train_model(train_config, edl_model, data_train, data_validate, model_type)

# Evaluate Model
test_signal_data = preproc.get_signal_data(train_config)
model_evaluator.evaluate_model(train_config, trained_model, data_test, test_signal_data, model_type)


# Plot Latent Space Information This is not Required for Model training and Analysis
# Here we Plot the Uncertainty on individual Plots
# Background
uncertainty_list = []
norm_signal_data = data_preprocessor.get_normalized_signals(train_config, test_signal_data)
alpha, z = trained_model.encoder.predict(data_test)
probs_background, S, uncertainty_background = loss_calculator.calculate_probability_uncertainity(alpha)
latent_space_graph_generator.generate_probability_distribution_hist(train_config, probs_background, "Background")
uncertainty_list.append(uncertainty_background)

# Signal
labels = train_config["data_files"]["signal_labels"]
signal_probability_list = []
for i in range(len(norm_signal_data)):
    alpha, z = trained_model.encoder.predict(norm_signal_data[i])
    probs_signal, S, uncertainty_signal = loss_calculator.calculate_probability_uncertainity(alpha)
    signal_probability_list.append(probs_signal)
    uncertainty_list.append(uncertainty_signal)
    latent_space_graph_generator.generate_probability_distribution_hist(train_config, probs_signal, labels[i])

# Plot Latent Space for all of the Uncertainities on one plot
latent_space_graph_generator.uncertainty_distribution_hist(train_config, uncertainty_list, ["Background"]+labels)

