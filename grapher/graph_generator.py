import os
import datetime
import numpy as np
import sklearn.metrics as sk
import matplotlib.pyplot as plt
import tensorflow as tf
import keras


def generate_loss_vs_epoch(train_config, history, model_type):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE) (Log)')
    plt.title("Loss vs epoch curve for " + train_config["model_files"]["model_name"])
    plt.xticks(np.arange(0,21,2))
    plt.plot(np.linspace(0, train_config["hyperparameters"]["epochs"], train_config["hyperparameters"]["epochs"]), loss)
    plt.plot(np.linspace(0, train_config["hyperparameters"]["epochs"], train_config["hyperparameters"]["epochs"]), val_loss)
    if model_type == 'EDL' or model_type =='VAE':
        plt.yscale('log')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(train_config["outputs"]["graph_directory"], datetime.datetime.now().strftime(
        train_config["outputs"]["date_time_format"]) + "_" + model_type + "_LossVsEpoch.png"), format="png", bbox_inches="tight")
    plt.show()


def generate_roc_curve(train_config, truth_values, scores ):
    fpr, tpr, _ = sk.roc_curve(truth_values, scores)
    roc_auc = sk.auc(fpr, tpr)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='darkorange')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('benchmark_ae_roc_curve_5732163')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(train_config["outputs"]["graph_directory"], datetime.datetime.now().strftime(
        train_config["outputs"]["date_time_format"]) + "_ROC.png"), format="png", bbox_inches="tight")
    plt.show()
