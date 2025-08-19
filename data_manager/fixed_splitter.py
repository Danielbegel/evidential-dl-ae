import numpy as np

"""
Here based on the prefixed values we will be generating splits.
"""



def generate_fixed_trainingdata_split(train_config, data, all_data=True):
    
    # generate training set 
    data_train = data[train_config["data_split"]["train_split_start"]:train_config["data_split"]["train_split_end"], :, 0:3].reshape(-1, 57) 



    # generate validation set
    data_validate = data[train_config["data_split"]["val_split_start"]:train_config["data_split"]["val_split_end"], :, 0:3].reshape(-1, 57) 




    if all_data:
        data_test = data[train_config["data_split"]["test_split_start"]:train_config["data_split"]["test_split_end"], :,
                    0:3].reshape(-1, 57)
        
        # Truncate test data if specified in config
        if train_config["data_split"]["max_data"] != "None":
            data_test = data_test[:train_config["data_split"]["max_data"]]
        


        print("Generated The Following Datasets\n Train Data Shape : {}\n Validation Data Shape : {}\n Test Data Shape : {}".format(data_train.shape, data_validate.shape,data_test.shape))
        return data_train, data_validate, data_test
    


    print("Generated The Following Datasets\n Train Data Shape : {}\n Validation Data Shape : {}\n Test Data Shape : {}".format(data_train.shape, data_validate.shape,data_test.shape))
    return data_train, data_validate
