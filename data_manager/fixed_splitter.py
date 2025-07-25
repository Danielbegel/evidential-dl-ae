
def generate_fixed_trainingdata_split(train_config, data, all_data=True):
    """
    Here based on the prefixed values we will be generating splits
    """
    for i, batch in enumerate(data):
        pt_sum = 0
        for j, particle in enumerate(data[i, :, :]):
            if particle[3] != 0:
                pt_sum += particle[0]
        for j, particle in enumerate(data[i, :, :]):
            particle[0] = particle[0] / pt_sum
    data_train = data[train_config["data_split"]["train_split_start"]:train_config["data_split"]["train_split_end"], :, 0:3].reshape(-1, 57)
    data_validate = data[train_config["data_split"]["val_split_start"]:train_config["data_split"]["val_split_end"], :, 0:3].reshape(-1, 57)
    if all_data:
        data_test = data[train_config["data_split"]["test_split_start"]:train_config["data_split"]["test_split_end"], :,
                    0:3].reshape(-1, 57)
        if train_config["data_split"]["max_data"] != "None":
            data_test = data_test[:train_config["data_split"]["max_data"]]
        print("Generated The Following Datasets\n Train Data Shape : {}\n Validation Data Shape : {}\n Test Data Shape : {}".format(data_train.shape, data_validate.shape,data_test.shape))
        return data_train, data_validate, data_test
    print("Generated The Following Datasets\n Train Data Shape : {}\n Validation Data Shape : {}\n Test Data Shape : {}".format(data_train.shape, data_validate.shape,data_test.shape))
    return data_train, data_validate


def generate_fixed_testdata_split(train_config, data):
    """
    This generates a test split based on the prefixed values
    """
    data_test = data[train_config["data_split"]["test_split_start"]:train_config["data_split"]["test_split_end"], :, 0:3]
    return data_test
