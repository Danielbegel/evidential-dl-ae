from model_manager import model_generator, model_trainer, model_storage
import json
from data_manager import data_preprocessor as preproc
from data_manager import fixed_splitter as fs
from grapher import graph_generator as gg



'''Open train config...'''
print("Opening train config file...")
TRAINER_CONFIG_FILE_LOCATION = 'config/trainer_config.json'
with open(TRAINER_CONFIG_FILE_LOCATION, 'r') as file:
    train_config = json.load(file)

model_type = train_config["model_files"]["model_type"]



print("preprocessing background data...")
background_data = preproc.get_data(train_config) # get background data
print("normalizing background data...")
background_data = preproc.normalize_data(background_data) # normalize background data
print("preprocessing signal data...")
signal_data = preproc.get_signal_data(train_config) # get signal data
print("normalizing signal data...")
signal_data = preproc.normalize_signal_data(signal_data) # normalize signal data



# split bg data into train, validate, and test sets.
print("splitting background data...")
data_train, data_validate, data_test = fs.generate_fixed_trainingdata_split(train_config, background_data, True)



# create model
print("initializing model...")
model = model_generator.create_model(train_config, model_type)



# train model
print("training model...")
model, history = model_trainer.train_model(train_config, model, data_train, data_validate, model_type)



gg.generate_loss_vs_epoch(train_config, history, model_type) # create loss vs epoch plots
gg.generate_loss_histogram(train_config, model, data_test, signal_data) # create loss histogram
gg.generate_roc_curve(train_config, model, data_test, signal_data) # create roc curve



# save the model
model_storage.save_model(train_config, model) 