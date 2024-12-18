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
import torch
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.optim import lr_scheduler
from CSST_Net import CSST_Net
from process import convert, save
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from visualize import draw_Tr_losses, draw_Te_losses, draw_Te_acc, draw_Tr_acc, MLP_T_SNE,draw_LR, plot_confusion_matrix
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

color_start = '\033[36m'
color_end = '\033[0m'
def train(Model,Tr_data,Tr_label,Te_data,Te_label):
    if LOAD_LOSS_ACC:
        Tr_Losses = np.load('Loss_Acc/Tr_loss.npy')
        Te_Losses = np.load('Loss_Acc/Te_loss.npy')
        Tr_Accuracy = np.load('Loss_Acc/Tr_acc.npy')
        Te_Accuracy = np.load('Loss_Acc/Te_acc.npy')
        Tr_Losses = Tr_Losses.tolist()
        Te_Losses = Te_Losses.tolist()
        Tr_Accuracy = Tr_Accuracy.tolist()
        Te_Accuracy = Te_Accuracy.tolist()
    else:
        Tr_Losses = []
        Te_Losses = []
        Tr_Accuracy = []
        Te_Accuracy = []
    if LOAD_WEIGHT:
        Model.load_state_dict(torch.load(PRE_TRAINING))
    "================Parameter configuration==========================="
    best_accuracy = 0.0
    best_loss = 1
    best_model_weights = None
    beat_epoch = 0
    current_time = 0
    learning_rates = []
    "====================Data loading============================="
    dataset = TensorDataset(Tr_data, Tr_label)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(Model.parameters(), lr=LR)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH)
    loss_func = nn.CrossEntropyLoss().to(device)

    for epoch in range(1,EPOCH+1):
        Model.train()
        tqdm_loader = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCH}",bar_format=color_start + '{l_bar}{bar}{r_bar}' + color_end, unit="batch")
        for step, (batch_X,batch_Y) in enumerate(tqdm_loader):
            Model.train()
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            Tr_output = Model(batch_X)
            Tr_loss = loss_func(Tr_output, batch_Y)
            optimizer.zero_grad()
            Tr_loss.backward()
            optimizer.step()
            if step == len(tqdm_loader)-1:
                Model.eval()
                with torch.no_grad():
                    num_samples = 240
                    Tr_indices = torch.randperm(len(Tr_data))[:num_samples]
                    sampled_Tr_data = Tr_data[Tr_indices]
                    sampled_Tr_label = Tr_label[Tr_indices]
                    sampled_Tr_data, sampled_Tr_label = sampled_Tr_data.to(device), sampled_Tr_label.to(device)

                    Te_indices = torch.randperm(len(Te_data))[:num_samples]
                    sampled_Te_data = Te_data[Te_indices]
                    sampled_Te_label = Te_label[Te_indices]
                    sampled_Te_data, sampled_Te_label = sampled_Te_data.to(device), sampled_Te_label.to(device)

                    Tr_output = Model(sampled_Tr_data)
                    Tr_loss = loss_func(Tr_output, sampled_Tr_label)
                    Te_output = Model(sampled_Te_data)
                    Te_loss = loss_func(Te_output, sampled_Te_label)

                    Tr_Losses.append(Tr_loss.item())
                    Te_Losses.append(Te_loss.item())

                    Tr_acc = ACC(sampled_Tr_data, sampled_Tr_label)
                    Te_acc = ACC(sampled_Te_data, sampled_Te_label)

                    Tr_Accuracy.append(Tr_acc)
                    Te_Accuracy.append(Te_acc)

                    tqdm_loader.set_postfix(Tr_loss = Tr_loss.item(),Te_loss = Te_loss.item(),Tr_acc = Tr_acc,Te_acc = Te_acc)

                    "====================================================================="
                    if Te_acc >= best_accuracy and Te_loss.item() <= best_loss:
                        best_accuracy = Te_acc
                        best_loss = Te_loss.item()
                        best_model_weights = Model.state_dict()  # Data loading saves the weight of the model with the highest accuracy
                        beat_epoch = epoch
                        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        learning_rates.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    "===========================TSNE======================"
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

    "==================Weight============================="
    Data_path = os.path.join(WEIGHTS, f'Model_{beat_epoch}_{current_time}.pth')
    torch.save(best_model_weights, Data_path)
    print("The training weights are saved to the directory: ", Data_path)

    "=================================================="
    torch.cuda.empty_cache()

    "===================Save============================"
    save(Tr_Losses, Te_Losses, Tr_Accuracy, Te_Accuracy)

    "===================Draw============================"
    draw_Tr_losses(Tr_Losses)
    draw_Te_losses(Te_Losses)
    draw_Tr_acc(Tr_Accuracy)
    draw_Te_acc(Te_Accuracy)
    draw_LR(learning_rates)

def ACC(data,label):
    y_pre = Model(data)
    label_pre = y_pre.argmax(dim=1)
    label_true = label.argmax(axis=1)
    correct = (label_pre == label_true).sum().item()
    total = label_true.size(0)
    accuracy = correct / total
    return accuracy

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
parser.add_argument("--epoch", type = int, default = 30, help = "Number of training iterations")
parser.add_argument("--batch_size", type = int, default = 32, help = "Batch")
parser.add_argument("--learning_rate", type = float, default = 0.0004, help = "learning rate")
parser.add_argument("--train_txt", type =  str, default = r'predata', help = "Data path")
parser.add_argument("--pre_training_weight", type = str, default = "", help = "Pre-train the weight path")
parser.add_argument("--load_weight", type = bool, default = 0, help = "Whether to load the pre-training weight")
parser.add_argument("--load_loss_acc", type = bool, default = False, help = "Whether to load the last training result")
parser.add_argument("--weights", type = str, default = "./weights/", help ="Trained weight saving path")
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
EPOCH = opt.epoch
BATCH_SIZE = opt.batch_size
LR = opt.learning_rate
DATA_PATH = opt.train_txt
PRE_TRAINING = opt.pre_training_weight
LOAD_WEIGHT = opt.load_weight
LOAD_LOSS_ACC= opt.load_loss_acc
WEIGHTS = opt.weights

"Data preprocessing"
data = np.load(DATA_PATH + '/' + 'data_6.npy')
label = np.load(DATA_PATH + '/' + 'label_6.npy')

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
train(Model,Tr_data=data_train,Tr_label=label_train,Te_data=data_test,Te_label=label_test)

