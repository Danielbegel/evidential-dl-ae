import json
from data_splitter import fixed_splitter as fs
from data_preprocessor import data_preprocessor as preproc
from model_manager import model_evaluation, ae_model_generator
from grapher import loss_graph_generator
import tensorflow as tf
import keras
from keras import layers

print("Initializing Configuration for Training...")
TRAINER_CONFIG_FILE = 'config/trainer_config_v1.0.json'

# Initialize the Config here
with open(TRAINER_CONFIG_FILE, 'r') as file:
    train_config = json.load(file)

print("Configuration Initialized")

# Get Data
print("Preprocessing Data")
preprocessed_data = preproc.get_data(train_config)
signal_data = preproc.get_signal_data(train_config)

# Create Split
print("Splitting Data")
data_train, data_validate, data_test = fs.generate_fixed_trainingdata_split(train_config, preprocessed_data, True)

# Load previous Model
print("Loading Model")
model = tf.keras.models.load_model("models/benchmark_ae_3.keras", custom_objects={'AeModel': ae_model_generator})

# Evaluate Model
print("Evaluating Model")
results = model_evaluation.evaluate_model(train_config, model)