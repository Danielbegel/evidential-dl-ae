import h5py
import numpy as np


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
    data_file = h5py.File(train_config["data_files"][file_type], 'r')
    if not is_file_valid(train_config, data_file):
        raise Exception("File Does not contain the required dataset")
    preprocessed_data = np.array(data_file["Particles"])
    return preprocessed_data


def get_signal_data(train_config):
    """
    This Function gets the signal data from the files specified in the config file
    """
    signal_list = train_config["data_files"]["signal_file_list"]
    data_list = []
    for file in signal_list:
        if train_config["data_split"]["max_data"] == "None":
            data_list.append(get_data(train_config, file_type=file))
        else:
            data_list.append(get_data(train_config, file_type=file)[:train_config["data_split"]["max_data"]])
    return data_list


def get_normalized_signals(train_config, signal_data):
    """
    Normalize Signal Values, Note that this is a Default Method used in the original notebook and has been kept the same
    """
    signal_labels = train_config["data_files"]["signal_labels"]
    for k, subset in enumerate(signal_data):
        for i, batch in enumerate(subset):
            pt_sum = 0
            for j, particle in enumerate(subset[i, :, :]):
                if particle[3] != 0:
                    pt_sum += particle[0]
            for j, particle in enumerate(subset[i, :, :]):
                particle[0] = particle[0] / pt_sum
        print("Loaded Signal " + signal_labels[k])

    normed_signals = []
    for j, subset in enumerate(signal_data):
        normed_signals += [np.reshape(subset[:, :, 0:3], (-1, 57))]
    return normed_signals
