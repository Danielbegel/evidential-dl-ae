# Imports
import os
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np


def train_model(train_config, model, data_train, data_validate, model_type="AE"):
    if model_type == "AE" or model_type == "VAE" or model_type == "EDL":
        return train_model_generic(train_config, model_type, model, data_train, data_validate)
    # Temp
    if model_type == 'Custom_Classification_Model':
        return train_model_classification(train_config, model, data_train, data_validate)


def train_model_generic(train_config, model_type, model, data_train, data_validate):
    """
    This function trains the models and logs the training process. The details for training can also be viewed in tensorboard
    """
    training_logger = CSVLogger(os.path.join(train_config["logger"]["training_log_filepath"] +
                                             train_config["model_files"]["model_name"],
                                             train_config["logger"]["training_log_file_suffix"]))
    tensorboard_callback = TensorBoard(log_dir=os.path.join(train_config["logger"]["training_log_filepath"] +
                                                            train_config["model_files"]["model_name"],
                                                            train_config["logger"]["tensorboard_log_subfilepath"]))
    early_stopping = EarlyStopping(patience=train_config["hyperparameters"]["stop_patience"], restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  patience=train_config["hyperparameters"]["lr_patience"], verbose=1)


    if model_type == "AE":
        print("training AE model")
        model.compile(optimizer=Adam(), loss=tf.keras.losses.MeanSquaredError())
        history = model.fit(x=data_train,
                        y=data_train,
                        validation_data=(data_validate, data_validate),
                        epochs=train_config["hyperparameters"]["epochs"],
                        batch_size=train_config["hyperparameters"]["batch_size"],
                        callbacks=[early_stopping, reduce_lr, tensorboard_callback, training_logger]
                        )
    elif model_type == "EDL":
        kl_loss = model.get_kl_loss()
        kl_scheduler = KLStrengthScheduler(schedule_fn=kl_schedule)
        one_hot_labels_train = np.column_stack((np.zeros(len(data_train)),np.ones(len(data_train))))
        one_hot_labels_val = np.column_stack((np.zeros(len(data_validate)),np.ones(len(data_validate))))

        print(one_hot_labels_train.shape)
        print(one_hot_labels_val.shape)

        print("training EDL model")
        model.compile(optimizer=Adam(), metrics=kl_loss)
        x_train = (data_train, one_hot_labels_train)
        y_train = one_hot_labels_train
        x_val = (data_validate, one_hot_labels_val)
        y_val = one_hot_labels_val
        history = model.fit(
                    x=x_train,
                    y=y_train,  
                    validation_data=(x_val,y_val),
                    epochs=train_config["hyperparameters"]["epochs"],
                    batch_size=train_config["hyperparameters"]["batch_size"],
                    callbacks=[early_stopping, reduce_lr, tensorboard_callback, training_logger, kl_scheduler]
                    )
    elif model_type == "VAE":
        kl_loss = model.get_kl_loss()
        model.compile(optimizer=Adam(), metrics=kl_loss)
        x_train = data_train
        y_train = data_train
        x_val = data_validate
        y_val = data_validate
        history = model.fit(
                    x=x_train,
                    y=y_train,
                    validation_data=(x_val, y_val),
                    epochs=train_config["hyperparameters"]["epochs"],
                    batch_size=train_config["hyperparameters"]["batch_size"],
                    callbacks=[early_stopping, reduce_lr, tensorboard_callback, training_logger]
                    )
            
    else:
        model.compile(optimizer=Adam())

    return model, history

def train_model_classification(train_config, model, data_train, label_data):
    """
    This function trains the models and logs the training process. The details for training can also be viewed in tensorboard
    """
    training_logger = CSVLogger(os.path.join(train_config["logger"]["training_log_filepath"] +
                                             train_config["model_files"]["model_name"],
                                             train_config["logger"]["training_log_file_suffix"]))
    tensorboard_callback = TensorBoard(log_dir=os.path.join(train_config["logger"]["training_log_filepath"] +
                                                            train_config["model_files"]["model_name"],
                                                            train_config["logger"]["tensorboard_log_subfilepath"]))
    early_stopping = EarlyStopping(patience=train_config["hyperparameters"]["stop_patience"], restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  patience=train_config["hyperparameters"]["lr_patience"], verbose=1)
    if train_config["hyperparameters"]["optimizer"] == 'adam':
        model.compile(optimizer=Adam())
    else:
        model.compile()
    history = model.fit(x=data_train,
                        y=label_data,
                        epochs=train_config["hyperparameters"]["epochs"],
                        batch_size=train_config["hyperparameters"]["batch_size"],
                        callbacks=[early_stopping, reduce_lr, training_logger, tensorboard_callback])
    return model, history



class KLStrengthScheduler(tf.keras.callbacks.Callback):
    def __init__(self, schedule_fn):
        super().__init__()
        self.schedule_fn = schedule_fn

    def on_epoch_end(self, epoch, logs=None):
        new_strength = self.schedule_fn(epoch)
        self.model.kl_strength = new_strength
        print(f"\n[Callback] Epoch {epoch+1}: kl_strength updated to {new_strength:.6f}")

def kl_schedule(epoch):
    # linearly increase to 1.0 over first 10 epochs
    return min(1.0, 0.1 * (epoch + 1))