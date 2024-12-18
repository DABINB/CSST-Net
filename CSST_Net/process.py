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
import torch
import numpy as np

def convert(data_train, data_test, label_train, label_test):
    data_train = torch.tensor(data_train)
    data_test = torch.tensor(data_test)
    label_train = torch.tensor(label_train)
    label_test = torch.tensor(label_test)

    data_train = data_train.double()
    data_test = data_test.double()
    label_train = label_train.double()
    label_test = label_test.double()
    return data_train, data_test, label_train, label_test

def save(Tr_Losses,Te_Losses,Tr_Accuracy,Te_Accuracy):
    Tr_loss = np.array(Tr_Losses)
    Te_loss = np.array(Te_Losses)
    Tr_acc = np.array(Tr_Accuracy)
    Te_acc = np.array(Te_Accuracy)
    np.save('Loss_Acc/Tr_loss.npy', Tr_loss)
    np.save('Loss_Acc/Te_loss.npy', Te_loss)
    np.save('Loss_Acc/Tr_acc.npy', Tr_acc)
    np.save('Loss_Acc/Te_acc.npy', Te_acc)
















