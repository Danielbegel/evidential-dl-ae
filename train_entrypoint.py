# This is sample code that shows how this code can be used

# Imports
import json
from data_splitter import fixed_splitter as fs
from data_preprocessor import data_preprocessor as preproc
from model_manager import model_trainer, model_generator, model_storage

print("Initializing Configuration for Training...")
TRAINER_CONFIG_FILE = 'config/trainer_config_v1.0.json'

# Initialize the Config here
with open(TRAINER_CONFIG_FILE, 'r') as file:
    train_config = json.load(file)

print("Configuration Initialized")

# Get Data
print("Preprocessing Data")
preprocessed_data = preproc.get_data(train_config)

# Create Split
print("Splitting Data")
data_train, data_validate = fs.generate_fixed_trainingdata_split(train_config, preprocessed_data)

# Create Model
print("Creating Model")
model = model_generator.create_model(train_config, 'AE')
#model = model_generator.create_model(train_config, 'EDL')

# Train the Model
print("Training Model")
trained_model = model_trainer.train_model(train_config, model, data_train, data_validate)

# Store the Model generated
model_storage.save_model(train_config, trained_model)
