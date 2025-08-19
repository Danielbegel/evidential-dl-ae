import numpy as np
import h5py
import tensorflow as tf

"""
This file contains functions for fetching background and signal data from .h5 files 
and functions for normalizing data before splitting
"""

def is_file_valid(train_config, data_file):
    """
    This function validates if the files contain the necessary columns
    """
    required_dataset = train_config["datasets"]["required_dataset"]
    for key in data_file['/']:
        if isinstance(data_file['/' + key], h5py.Dataset) and key == required_dataset:
            return True
    return False





def get_data(train_config, file_type="background_file"):
    """
    This function can pull the data from multiple sources, both background and signal we have set the default to a
    background file here
    """
    
    required_dataset = train_config["datasets"]["required_dataset"]
    data_file = h5py.File(train_config["data_files"][file_type], 'r')
    if not is_file_valid(train_config, data_file):
        raise Exception("File Does not contain the required dataset")
    preprocessed_data = np.array(data_file[required_dataset])
    return preprocessed_data





def get_signal_data(train_config):
    """
    This Function gets the signal data from the files specified in the config file
    """

    signal_name_list = train_config["data_files"]["signal_file_list"]
    signal_list = []
    for signal_file in signal_name_list:
        if train_config["data_split"]["max_data"] == "None":
            signal_list.append(get_data(train_config, file_type=signal_file))
        else:
            signal_list.append(get_data(train_config, file_type=signal_file)[:train_config["data_split"]["max_data"]])
    return signal_list





def normalize_data(data):
    """
    Normalizes data
    """
    for i, batch in enumerate(data):
        pt_sum = 0
        for j, particle in enumerate(data[i, :, :]):
            if particle[3] != 0:
                pt_sum += particle[0]
        for j, particle in enumerate(data[i, :, :]):
            particle[0] = particle[0] / pt_sum
    return data





def normalize_signal_data(data):
    """
    Normalizes a set of signal data
    """
    normalized_data = []
    for data_file in data:
        normalize_data(data_file)
        normalized_data.append(data_file[:, :,0:3].reshape(-1, 57))
    return normalized_data
