# **[TIM] A High-Precision Aeroengine Bearing Fault Diagnosis Based on Spatial Enhancement Convolution and Vision Transformer**🥳
**This is the official PyTorch codes for the paper:**
> [Bin Wang](https://github.com/DABINB), Yongcheng Xiong, [Liguo Tan*](https://homepage.hit.edu.cn/tanliguo?lang=zh). A high-precision aeroengine bearing fault diagnosis based on spatial enhancement convolution and vision transformer[J]. IEEE Transactions on Instrumentation and Measurement, 2025, 74: 1-15.
# Network Architecture 💐
![CSST-Net](https://github.com/user-attachments/assets/c88b8594-66a6-40de-b1aa-6dc1e9096408)

# News 🚀
- Dec 18, 2024: We release training code.


# Getting started
## Install
We test the code on Pytorch 2.2.1 + CUDA 11.8
1. Create a new conda environment, or take advantage of an existing conda environment.
> [!NOTE]
> Make sure python version 3.9 is available.

```
conda create -n CSSTNet python=3.9
conda activate CSSTNet
```
2. Install dependencies
```
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```
# Training and Evaluation
## Prepare dataset
You can download the datasets on [Google Drive](https://drive.google.com/drive/folders/1Km1Go4ilB_bI033SBJ7eJ0uCzbqEqbgt)。

Then arrange the data in the following format：
```
|-data
  |-data1.npy
  |-data2.npy
  |-data3.npy
  |-data4.npy
  |-data5.npy
```
Once you have downloaded the data set and settled it according to the above requirements, run the following statement to make the data you need for training and testing.



```
python dataset.py
```
> [!CAUTION]
> The current code can only be used to make noisy data, if you make noise-free data, please manually change the code.

After production, the predata folder will contain the following files:
```
|-predata
  |-data.npy
  |-label.npy
  |-data_0.npy
  |-label_0.npy
  |-data_2.npy
  |-label_2.npy
  |-data_6.npy
  |-label_6.npy
  |-data_10.npy
  |-label_10.npy
  |-data_-2.npy
  |-label_-2.npy
```

## Train
Once the data set is ready, training can be performed

```
python train.py
```

## Test
Please match the pre-trained weights with the dataset for testing, then:

```
python test.py
```

# Citation  💓
If you find our work useful for your research, please cite us:

```
@ARTICLE{10758748,
  author={Wang, Bin and Xiong, Yongcheng and Tan, Liguo},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={A High-Precision Aeroengine Bearing Fault Diagnosis Based on Spatial Enhancement Convolution and Vision Transformer}, 
  year={2025},
  volume={74},
  number={},
  pages={1-15},
  keywords={Feature extraction;Fault diagnosis;Aircraft propulsion;Accuracy;Noise;Convolution;Time-frequency analysis;Transforms;Transformers;Computer vision;Aeroengine intershaft bearing;convolutional neural network (CNN);multisensor information fusion;spatial enhancement;vision transformer (VIT)},
  doi={10.1109/TIM.2024.3502884}}
```

# Contact ☺️
If you have any questions, please feel free to contact the author.

Bin Wang: [23s104106@stu.hit.edu.cn](23s104106@stu.hit.edu.cn)
