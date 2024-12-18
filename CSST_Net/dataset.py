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
import pywt
import torch
import matplotlib
import numpy as np
from skimage.transform import resize
matplotlib.rc("font", family='Microsoft YaHei')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataProcess():
    def __init__(self, data_path):
        super(DataProcess, self).__init__()
        self.window_size = 1024
        self.num_classe = 4
        self.totalscale = 256
        self.sampling_period = 1/25000
        self.wavename = 'cmorl3-3'
        self.path = data_path
        self.data, self.label = self.get_data_label()

    def get_data_label(self):
        file_list = os.listdir(self.path)
        SNR = [-2, 0, 2, 6, 10]
        for j in range(len(SNR)):
            alldata = []
            alllabel = []
            for i in range(len(file_list)):
                if file_list[i].endswith('1.npy') or file_list[i].endswith('3.npy') or file_list[i].endswith('4.npy') or file_list[i].endswith('5.npy'):
                    file = self.path+'/'+file_list[i]
                    signal = self.Takeout(file)
                    noise_singal = self.add_white_noise_with_snr(signal, SNR[j])
                    temp = self.normalization(noise_singal)
                    data = self.sliding_window(temp, self.window_size)
                    indices = np.arange(0, 7200, 36)
                    sampled_data = data[indices]
                    makedata, makelabel = self.makedata(sampled_data,file)
                    if i == 0:
                        alldata = makedata
                        alllabel = makelabel
                    else:
                        alldata = np.vstack([alldata, makedata])
                        alllabel = np.vstack([alllabel, makelabel])
            alldata = np.array(alldata)
            alllabel = np.array(alllabel)
            print(alldata.shape)
            print(alllabel.shape)
            np.save(f'predata\data_{SNR[j]}.npy', alldata)
            np.save(f'predata\label_{SNR[j]}.npy', alllabel)
        return alldata, alllabel

    def makedata(self,dataset,file):
        if '1.npy' in file:
            text = 'normal'
        elif '3.npy' in file:
            text = "inner1"
        elif '4.npy' in file:
            text = "inner2"
        elif '5.npy' in file:
            text = "outer"
        makedatas = []
        makelabels = []
        for i in range(len(dataset)):
            group_data = []
            data_list = dataset[i]
            for j in range(data_list.shape[0]):
                fc = pywt.central_frequency(self.wavename)
                cparam = 2 * fc * self.totalscale
                scales = cparam / np.arange(self.totalscale, 0, -1)
                coefficients, frequencies = pywt.cwt(data_list[j], scales, self.wavename, self.sampling_period)
                amp = abs(coefficients)
                resized_image = resize(amp, (224, 224), anti_aliasing=True)
                resized_image = np.array(resized_image)
                group_data.append(resized_image)
            makedatas.append(group_data)
            if text == 'normal':
                makelabels.append([1, 0, 0, 0])
            elif text == 'inner1':
                makelabels.append([0, 1, 0, 0])
            elif text == 'inner2':
                makelabels.append([0, 0, 1, 0])
            elif text == 'outer':
                makelabels.append([0, 0, 0, 1])
        return np.array(makedatas), np.array(makelabels)

    def sliding_window(self, signal, window_size):
        step = window_size
        windows = []
        for i in range(20):
            for start in range(0, signal.shape[2] - window_size + 1, step):
                end = start + window_size
                window = signal[i,:,start:end]
                windows.append(window)
        return np.array(windows)

    def Takeout(self,file):
        dataset = np.load(file)
        LP = [1000, 1500, 2000, 2500, 3000, 3500, 3600, 3800, 4000, 4200, 4400, 4500, 4600, 5000, 3000, 3000, 3000, 3000, 3000, 3000]
        HP = [1200, 1800, 2400, 3000, 3600, 4200, 4320, 4560, 4800, 5040, 5280, 5400, 5520, 6000, 3900, 4200, 4500, 4800, 5100, 5400]
        datas = []
        for j in range(20):
            num = 0
            acc1, acc2, acc4 = [], [], []
            reshape_data_list = []
            for i in range(dataset.shape[0]):
                if dataset[i, 6, 0] == LP[j] and dataset[i, 6, 1] == HP[j]:
                    num += 1
                    acc1.append(dataset[i, 2, :])
                    acc2.append(dataset[i, 3, :])
                    acc4.append(dataset[i, 5, :])
                    if num == 18:
                        break
                else:
                    continue
            data_list = [acc1, acc2, acc4]
            for list in data_list:
                reshape_data_list.append(np.array(list).flatten())
            datas.append(reshape_data_list)
        return np.array(datas)

    def normalization(self, noise_signal):
        normdatas = []
        for i in range(len(noise_signal)):
            mean_dict = []
            std_dict = []
            norm_data = []
            for relist in noise_signal[i]:
                mean = np.mean(relist)
                std = np.std(relist)
                mean_dict.append(mean)
                std_dict.append(std)
            for i, reshapedata in enumerate(noise_signal[i]):
                reshaped_data = (reshapedata - mean_dict[i])/std_dict[i]
                norm_data.append(reshaped_data)
            normdatas.append(norm_data)
        return np.array(normdatas)

    def add_white_noise_with_snr(self, signal, snr_db):
        noisedatas = []
        for i in range(len(signal)):
            noise_data = []
            for list in signal[i]:
                signal_power = np.sum(list** 2) / len(list)
                noise_power = signal_power / (10 ** (snr_db / 10))
                noise = np.random.normal(0, np.sqrt(noise_power), len(list))
                noisy_signal = list + noise
                noise_data.append(noisy_signal)
            noisedatas.append(noise_data)
        return np.array(noisedatas)


Dataset = DataProcess(data_path='data')
Data = Dataset.data
Label = Dataset.label
print(Data.shape)
print(Label.shape)
