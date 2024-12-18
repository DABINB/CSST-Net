"""
A High-Precision Aeroengine Bearing Fault Diagnosis Based on Spatial Enhancement Convolution and Vision Transformer
********************************************************************
*                                                                  *
* Copyright © 2024 All rights reserved                             *
* Written by Mr.Wangbin                                            *
* [December 18,2024]                                               *
*                                                                  *
********************************************************************
"""
import os
import itertools
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']

def draw_LR(lrs):
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, color='r')
    plt.text(0, lrs[0], str(lrs[0]))
    plt.show()

def VIT_T_SNE(VIT, label):
    folder_path = 'results'
    file_name = 'CNN_T_SNE.png'
    reshaped_data = VIT.view(VIT.size(0), -1).cpu().detach().numpy()
    labels = label.cpu().numpy()
    labels = np.argmax(labels, axis=1)
    #Data dimensionality reduction
    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(reshaped_data)
    tsne = TSNE(n_components=2, perplexity=100.0, n_iter=1000, random_state=42)
    tsne_result = tsne.fit_transform(pca_result)
    plt.figure(figsize=(8, 6))
    for i in range(4):
        indices = labels == i
        plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], label=f'Class {i}')
    plt.legend(fontsize=25)
    file_path = os.path.join(folder_path, file_name)
    plt.savefig(file_path)
    plt.show()

def MLP_T_SNE(MLP, label):
    folder_path = 'results'
    file_name = 'MLP_T_SNE.png'
    reshaped_data = MLP.squeeze().cpu().detach().numpy()
    labels = label.cpu().numpy()
    labels = np.argmax(labels, axis=1)
    tsne = TSNE(n_components=2, perplexity=30.0, n_iter=1000, random_state=42)
    tsne_result = tsne.fit_transform(reshaped_data)
    plt.figure(figsize=(8, 6))
    for i in range(4):
        indices = labels == i
        plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], label=f'Class {i}')
    plt.legend(fontsize=25)
    plt.xticks([])
    plt.yticks([])
    file_path = os.path.join(folder_path, file_name)
    plt.savefig(file_path)
    plt.show()

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues, normalize=False):
    folder_path = 'results'
    file_name = 'confusion_matrix.png'
    plt.imshow(cm, cmap=cmap)
    plt.title(title, fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm, 2)
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f') if normalize else cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=20)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    file_path = os.path.join(folder_path, file_name)
    plt.savefig(file_path)
    plt.show()


def draw_Tr_losses(Losses):
    folder_path = 'results'
    file_name = 'Tr_loss_curve.png'
    plt.plot(Losses)
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title('Training Loss', fontsize=20)
    file_path = os.path.join(folder_path, file_name)
    plt.savefig(file_path)
    plt.show()

def draw_Te_losses(Losses):
    folder_path = 'results'
    file_name = 'Te_loss_curve.png'
    plt.plot(Losses)
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title('Testing Loss', fontsize=20)
    file_path = os.path.join(folder_path, file_name)
    plt.savefig(file_path)
    plt.show()

def draw_Te_acc(Accuracy):
    folder_path = 'results'
    file_name = 'Te_acc_curve.png'
    plt.plot(Accuracy)
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Te_acc', fontsize=20)
    plt.title('Testing Accuracy', fontsize=20)
    file_path = os.path.join(folder_path, file_name)
    plt.savefig(file_path)
    plt.show()

def draw_Tr_acc(Accuracy):
    folder_path = 'results'
    file_name = 'Tr_acc_curve.png'
    plt.plot(Accuracy)
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Tr_acc', fontsize=20)
    plt.title('Training Accuracy', fontsize=20)
    file_path = os.path.join(folder_path, file_name)
    plt.savefig(file_path)
    plt.show()