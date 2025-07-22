import os
import datetime
import numpy as np
import sklearn.metrics as sk
import matplotlib.pyplot as plt


def generate_histogram(train_config, background_loss, signal_loss, labels):
    plt.hist(background_loss, histtype='step', bins=train_config["outputs"]["bins"], label='Background : {}'.format(len(background_loss)))
    for i, batch in enumerate(signal_loss):
        plt.hist(batch, bins=train_config["outputs"]["bins"], histtype='step', label=str(labels[i + 1]) + " : {}".format(len(batch)))
    plt.xlabel('Loss')
    plt.ylabel('Log Frequency')
    plt.yscale('log')
    plt.title("Loss Histogram")
    plt.legend(title=" Number of Events")
    plt.savefig(os.path.join(train_config["outputs"]["graph_directory"], datetime.datetime.now().strftime(
        train_config["outputs"]["date_time_format"]) + "_Histogram.png"), format="png", bbox_inches="tight")
    plt.show()


def generate_roc_curve(train_config, background_loss, signal_loss, labels):
    plt.plot(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), '--', label='Diagonal')
    for j, batch in enumerate(signal_loss):
        roc_data = np.concatenate((background_loss, batch))
        truth = []
        for i in range(len(background_loss)):
            truth += [0]
        for i in range(len(batch)):
            truth += [1]
        fpr, tpr, x = sk.roc_curve(truth, roc_data)
        auc = sk.roc_auc_score(truth, roc_data)
        plt.plot(fpr, tpr, label=labels[j + 1] + ": " + str(np.round(auc, 6)))
    plt.xlabel('FPR (Log)')
    plt.semilogx()
    plt.ylabel('TPR (Log)')
    plt.semilogy()
    plt.title("ROC for Model")
    plt.legend(title="AUC Scores")
    plt.savefig(os.path.join(train_config["outputs"]["graph_directory"], datetime.datetime.now().strftime(
        train_config["outputs"]["date_time_format"]) + "_ROC.png"), format="png", bbox_inches="tight")
    plt.show()

def generate_loss_vs_epoch_curve(train_config, loss, val_loss):
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title("Loss vs epoch curve for " + train_config["model_files"]["model_name"])
    plt.xticks(np.arange(0,21,2))
    plt.plot(np.linspace(0, train_config["hyperparameters"]["epochs"], train_config["hyperparameters"]["epochs"]), loss)
    plt.plot(np.linspace(0, train_config["hyperparameters"]["epochs"], train_config["hyperparameters"]["epochs"]), val_loss)

    plt.savefig(os.path.join(train_config["outputs"]["graph_directory"], datetime.datetime.now().strftime(
        train_config["outputs"]["date_time_format"]) + "_LossVsEpoch.png"), format="png", bbox_inches="tight")
    plt.show()