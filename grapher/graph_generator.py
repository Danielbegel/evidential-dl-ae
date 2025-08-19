import os
import datetime
import numpy as np
import sklearn.metrics as sk
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from model_manager import model_evaluation


def generate_loss_vs_epoch(train_config, history, model_type):
    '''
    Generate plot of the losses over time
    '''
    model_type = train_config["model_files"]["model_type"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = len(loss)

    if model_type == 'VAE' or model_type == 'EDL':
        fig, axs = plt.subplots(2,1, sharex=True)
        axs[0].plot(np.linspace(0, epochs, epochs), loss, label='training MSE')
        axs[0].plot(np.linspace(0, epochs, epochs), val_loss, label='validation MSE')
        axs[0].set_ylabel('loss')
        axs[0].legend(loc='upper right')
        axs[0].set_title('MSE loss vs epoch')


        kl_loss = history.history["kl_loss"]
        kl_val_loss = history.history["val_kl_loss"]
        axs[1].plot(np.linspace(0, epochs, epochs), kl_loss, label='training KL')
        axs[1].plot(np.linspace(0, epochs, epochs), kl_val_loss, label='validation KL')
        axs[1].set_ylabel('loss')
        axs[1].set_xlabel('epoch index')
        axs[1].set_title('KL loss vs epoch')
        axs[1].set_xticks(np.arange(0,epochs + 1,2))
        axs[1].legend(loc='upper right')


        fig.suptitle("Loss vs epoch curve for " + model_type)



    else:
        plt.plot(np.linspace(0, epochs, epochs), loss, label='training MSE')
        plt.plot(np.linspace(0, epochs, epochs), val_loss, label='validation MSE')
        plt.ylabel('loss')
        plt.legend(loc='upper right')
        plt.title('MSE loss vs epoch')

    
    plt.savefig(os.path.join(train_config["outputs"]["graph_directory"], datetime.datetime.now().strftime(
        train_config["outputs"]["date_time_format"]) + "_" + model_type + "_LossVsEpoch.png"), format="png", bbox_inches="tight")
    plt.tight_layout()
    plt.show()

def generate_roc_curve(train_config, model, background_data, signal_data):
    model_type = train_config["model_files"]["model_type"]
    if model_type == 'EDL' or model_type == 'VAE':
        mse, kl = model_evaluation.calculate_loss(train_config, model, signal_data[0], model_type)
        Ato4l_scores = mse + kl
        mse, kl = model_evaluation.calculate_loss(train_config, model, signal_data[1], model_type)
        hToTauTau_scores = mse + kl
        mse, kl = model_evaluation.calculate_loss(train_config, model, signal_data[2], model_type)
        hChToTauNu_scores = mse + kl
        mse, kl = model_evaluation.calculate_loss(train_config, model, signal_data[3], model_type)
        leptoquark_scores = mse + kl

        mse, kl = model_evaluation.calculate_loss(train_config, model, background_data, model_type)
        background_scores = mse + kl
    elif model_type == 'AE':
        Ato4l_scores = model_evaluation.calculate_loss(train_config, model, signal_data[0], model_type)
        hToTauTau_scores = model_evaluation.calculate_loss(train_config, model, signal_data[1], model_type)
        hChToTauNu_scores = model_evaluation.calculate_loss(train_config, model, signal_data[2], model_type)
        leptoquark_scores = model_evaluation.calculate_loss(train_config, model, signal_data[3], model_type)
        background_scores = model_evaluation.calculate_loss(train_config, model, background_data, model_type)

    else:
        raise Exception("Unsupported model type")
    
    
    signal_scores = np.stack((Ato4l_scores, hToTauTau_scores, hChToTauNu_scores, leptoquark_scores),axis=2)



    plt.figure()
    plt.plot(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), '--', label='Diagonal')
    for j in range(signal_scores.shape[2]):
        batch = signal_scores[:,:,j]
        roc_data = np.concatenate((background_scores, batch), axis=0)
        truth = []
        for i in range(len(background_scores)):
            truth += [0]
        for i in range(len(batch)):
            truth += [1]
        fpr, tpr, x = sk.roc_curve(truth, roc_data)
        auc = sk.roc_auc_score(truth, roc_data)
        plt.plot(fpr, tpr, label=train_config["data_files"]["signal_labels"][j] + ": " + str(np.round(auc, 6)))
    plt.xlabel('FPR (Log)')
    plt.semilogx()
    plt.ylabel('TPR (Log)')
    plt.semilogy()
    plt.title("ROC for benchmark " + model_type + " Model")
    plt.legend(title="AUC Scores")
    plt.savefig(os.path.join(train_config["outputs"]["graph_directory"], datetime.datetime.now().strftime(
            train_config["outputs"]["date_time_format"]) + "_" + model_type + "_ROC.png"), format="png", bbox_inches="tight")
    plt.show()

def generate_probability_distribution_hist(train_config, probs, label="Background"):
    anomaly_probs = probs[:, 1]
    plt.hist(
        anomaly_probs, 
        histtype='step', 
        bins=train_config["outputs"]["bins"], 
        label=label + ' probability distribution', 
    )
    plt.xlabel('Probability')
    plt.ylabel('Log Frequency')
    plt.yscale('log')
    plt.title("Probability distribution for " + label)
    plt.legend()
    # plt.savefig(os.path.join(train_config["outputs"]["graph_directory"], datetime.datetime.now().strftime(
    #         train_config["outputs"]["date_time_format"]) + "KL_Histogram.png"), format="png", bbox_inches="tight")
    plt.show()

def generate_uncertainty_distribution_hist(train_config, uncertainty_list, labels):
     
     for i, sublist in uncertainty_list:
        plt.hist(
            sublist, 
            histtype='step',
            bins=train_config["outputs"]["bins"], 
            label=labels[i] + ' uncertainty distribution', 
        )
     plt.xlabel('Probability')
     plt.ylabel('Log Frequency')
     plt.yscale('log')
     plt.title("Probability distribution for " + label)
     plt.legend()
     # plt.savefig(os.path.join(train_config["outputs"]["graph_directory"], datetime.datetime.now().strftime(
     # #         train_config["outputs"]["date_time_format"]) + "KL_Histogram.png"), format="png", bbox_inches="tight")
     plt.show()

def generate_loss_histogram(train_config, model, data_test, signal_data):
    model_type = train_config["model_files"]["model_type"]
    plt.figure()
    if model_type == "AE":
        for i, signal in enumerate(train_config["data_files"]["signal_file_list"]):
            mse = model_evaluation.calculate_loss(train_config, model, signal_data[i], model_type)
            plt.hist(mse, 
                    histtype='step', 
                    bins=train_config["outputs"]["bins"], 
                    label=train_config["data_files"]["signal_labels"][i] + f': {len(mse):.0f}', 
                    )

        mse_background = model_evaluation.calculate_loss(train_config, model, data_test, model_type)


        plt.hist(mse_background, histtype='step', bins=train_config["outputs"]["bins"], label=f"Background: {len(mse_background):.0f} ")
        plt.xlabel('Loss')
        plt.ylabel('Log Frequency')
        plt.yscale('log')
        plt.title("Loss Histogram for " + model_type)
        plt.legend(title=" Number of Events")
        plt.savefig(os.path.join(train_config["outputs"]["graph_directory"], datetime.datetime.now().strftime(
                train_config["outputs"]["date_time_format"]) + "MSE_Histogram.png"), format="png", bbox_inches="tight")
        plt.show()
    elif model_type == "EDL" or model_type == "VAE":
        for i, signal in enumerate(train_config["data_files"]["signal_file_list"]):
            mse, kl = model_evaluation.calculate_loss(train_config, model, signal_data[i], model_type)
            total_loss = mse + kl
            
            plt.hist(total_loss, 
                    histtype='step', 
                    bins=train_config["outputs"]["bins"], 
                    label=train_config["data_files"]["signal_labels"][i] + f': {len(mse):.0f}' 
            )

        mse_background, kl_background = model_evaluation.calculate_loss(train_config, model, data_test, model_type)
        total_loss = mse_background + kl_background

        plt.hist(total_loss, histtype='step', bins=train_config["outputs"]["bins"], label=f"Background: {len(mse_background):.0f} ")
        plt.xlabel('Loss')
        plt.ylabel('Log Frequency')
        plt.yscale('log')
        plt.title("Loss Histogram for " + model_type)
        plt.legend(title=" Number of Events")
        plt.savefig(os.path.join(train_config["outputs"]["graph_directory"], datetime.datetime.now().strftime(
                train_config["outputs"]["date_time_format"]) + model_type + "_loss_Histogram.png"), format="png", bbox_inches="tight")
        plt.show()
    else:
        raise NotImplemented