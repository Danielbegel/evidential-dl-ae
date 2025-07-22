import numpy as np
import h5py



def is_file_valid(train_config, data_file, required_dataset):
    """
    This function validates if the files contain the necessary columns
    """
    # required_dataset = train_config["datasets"]["required_dataset"]
    for key in data_file['/']:
        if isinstance(data_file['/' + key], h5py.Dataset) and key == required_dataset:
            return True
    return False


# def get_data(train_config, file_type="background_file"): DEBUG
def get_data(train_config, required_dataset, file_type="background_file"):
    """
    This function can pull the data from multiple sources, both background and signal we have set the default to a
    background file here
    """
    print(f"Attempting to open HDF5 file: {file_type}") ## DEBUG
    data_file = h5py.File(train_config["data_files"][file_type], 'r')
    # if not is_file_valid(train_config, data_file): DEBUG
    if not is_file_valid(train_config, data_file, required_dataset):
        raise Exception("File Does not contain the required dataset")
    # preprocessed_data = np.array(data_file["Particles"])
    preprocessed_data = np.array(data_file[required_dataset])
    if file_type == 'background_file':
        columns_to_keep = []
        selected_data = preprocessed_data[:,columns_to_keep]
    else:
        selected_data = preprocessed_data
    return selected_data


# def get_signal_data(train_config, file_name): #file_name the same thing as file_type but didnt want it to be confusing.
def get_signal_data(train_config, required_dataset, file_name): #file_name the same thing as file_type but didnt want it to be confusing.
    """
    This Function gets the signal data from the files specified in the config file
    """
    signal_list = train_config["data_files"]["signal_file_list"]
    data_list = []
    for file in signal_list:
        if train_config["data_split"]["max_data"] == "None":
            data_list.append(get_data(train_config, required_dataset, file_type=file_name))
        else:
            data_list.append(get_data(train_config, required_dataset, file_type=file_name)[:train_config["data_split"]["max_data"]])
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
