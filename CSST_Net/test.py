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
import argparse
import numpy as np
from CSST_Net import CSST_Net
from process import convert
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from visualize import  MLP_T_SNE,plot_confusion_matrix
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(Model,Te_data,Te_label):
    if LOAD_WEIGHT:
        Model.load_state_dict(torch.load(PRE_TRAINING))
    num_samples = 240
    Te_indices = torch.randperm(len(Te_data))[:num_samples]
    sampled_Te_data = Te_data[Te_indices]
    sampled_Te_label = Te_label[Te_indices]
    sampled_Te_data, sampled_Te_label = sampled_Te_data.to(device), sampled_Te_label.to(device)
    Model.eval()
    with torch.no_grad():
        Te_output = Model(sampled_Te_data)
    label_pre = Te_output.argmax(dim=1)
    label_true = sampled_Te_label.argmax(axis=1)
    correct = (label_pre == label_true).sum().item()
    total = label_true.size(0)
    accuracy = correct / total
    print("Accuracy：", accuracy)
    MLP_T_SNE(Te_output, sampled_Te_label)

    "==================Matrix============================="
    Te_output_cpu = Te_output.cpu()
    sampled_Te_label_cpu = sampled_Te_label.cpu()
    label_pre = Te_output_cpu.argmax(dim=1)
    label_true = sampled_Te_label_cpu.argmax(axis=1)
    confusion_mat = confusion_matrix(label_true,label_pre)
    plot_confusion_matrix(confusion_mat, classes = range(4))

    "==================conclusion=========================="
    conclusion(Te_output_cpu,sampled_Te_label_cpu)

    "=================================================="
    torch.cuda.empty_cache()

def conclusion(output,label):
    l_pre = output.argmax(dim=1)
    l_true = label.argmax(axis=1)
    report = classification_report(l_true,l_pre)
    print(report)

parser = argparse.ArgumentParser() #参数解析
parser.add_argument("--class_num", type = int, default = 4, help = "class")
parser.add_argument("--num_patches", type = int, default = 49, help = "Patch number")
parser.add_argument("--num_dim", type = int, default = 192, help = "Patch dimension")
parser.add_argument("--num_depth", type = int, default = 1, help = "Layers")
parser.add_argument("--num_head", type = int, default = 8, help = "Heads of attention")
parser.add_argument("--num_mlp_dim",  type = int, default = 768, help = "Hidden layer dimension")
parser.add_argument("--dropout", type = int, default = 0.1, help = "dropout rate")
parser.add_argument("--linear_dim", type = int, default = 64, help = "Fully connected hidden layer dimension")
parser.add_argument("--train_txt", type =  str, default = r'predata', help = "Data path")
parser.add_argument("--pre_training_weight", type = str, default = "weights\\None\Model_27_20240822-211300.pth", help = "Pre-train the weight path")
parser.add_argument("--load_weight", type = bool, default = True, help = "Whether to load the pre-training weight")
opt = parser.parse_args()
print(opt)

CLASS_NUM = opt.class_num
NUM_PATCHES = opt.num_patches
NUM_DIM = opt.num_dim
NUM_DEPTH = opt.num_depth
NUM_HEAD = opt.num_head
NUM_MLP_DIM = opt.num_mlp_dim
DROPOUT = opt.dropout
LINEAR_DIM = opt.linear_dim
DATA_PATH = opt.train_txt
PRE_TRAINING = opt.pre_training_weight
LOAD_WEIGHT = opt.load_weight

"Data preprocessing"
data = np.load(DATA_PATH + '/' + 'data.npy')
label = np.load(DATA_PATH + '/' + 'label.npy')

data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.3, random_state=42)
data_train, data_test, label_train, label_test = convert(data_train, data_test, label_train, label_test)  #

Model =CSST_Net(
    num_classes = CLASS_NUM,
    num_patches = NUM_PATCHES,
    dim = NUM_DIM,
    depth = NUM_DEPTH,
    heads = NUM_HEAD,
    mlp_dim = NUM_MLP_DIM,
    linear_dim = LINEAR_DIM,
    dropout = DROPOUT).double().to(device)

test(Model,Te_data=data_test,Te_label=label_test)

