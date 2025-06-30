# TMA AI Advance Group 51 - Plant Disease Detection
- This is a group project by a TMA team, carried out during the AI Advance course. The objective of the project is to leverage Convolutional Neural Networks (CNNs) to train on a dataset consisting of images of healthy and diseased plant leaves, and to deploy the trained model for detecting plant diseases based on leaf imagery.
- Team members:
  
  | No. | Name          |
  |-----|---------------|
  | 1   | Giang Nguyen  |
  | 2   | Vinh Nguyen   |
  | 3   | Nghiep Mai    |
  | 4   | Hiep Duong    |
  | 5   | Thinh Nguyen  |
  | 6   | Huy Luong     |
  | 7   | Do Uong       |
  | 8   | Tien Nguyen   |
  | 9   | Duc Tu        |

# Dataset
This project uses the PlantVillage dataset for training and evaluation.  
```bibtex
@article{Mohanty_Hughes_Salathé_2016,
title={Using deep learning for image-based plant disease detection},
volume={7},
DOI={10.3389/fpls.2016.01419},
journal={Frontiers in Plant Science},
author={Mohanty, Sharada P. and Hughes, David P. and Salathé, Marcel},
year={2016},
month={Sep}} 
```

## 1. Demo

https://github.com/user-attachments/assets/ef997f7d-4fbd-4e3e-be78-64062b878758


## 2. Training Results
### 2.1. ResNet18
| Train/Validation Accuracy | Train/Validation Loss |
|---------------------------|------------------------|
| ![Accuracy](https://github.com/huyluongme/TMA_AI_Advance_Group_51/blob/ec17b3713a7427f2f547dd80bf887d7514571555/checkpoint/resnet18_6/train/accuracy.png) | ![Loss](https://github.com/huyluongme/TMA_AI_Advance_Group_51/blob/ec17b3713a7427f2f547dd80bf887d7514571555/checkpoint/resnet18_6/train/loss.png) |

![Confusion Matrix](https://github.com/huyluongme/TMA_AI_Advance_Group_51/blob/ec17b3713a7427f2f547dd80bf887d7514571555/checkpoint/resnet18_6/test/confusion_matrix.png)
### 2.2. Mobilenet V2
| Train/Validation Accuracy | Train/Validation Loss |
|---------------------------|------------------------|
| ![Accuracy](https://github.com/huyluongme/TMA_AI_Advance_Group_51/blob/ec17b3713a7427f2f547dd80bf887d7514571555/checkpoint/mobilenet_1/train/accuracy.png) | ![Loss](https://github.com/huyluongme/TMA_AI_Advance_Group_51/blob/ec17b3713a7427f2f547dd80bf887d7514571555/checkpoint/mobilenet_1/train/loss.png) |

## 3. Installation
1. Install [Anaconda](https://www.anaconda.com/), Python and `git`.
2. Creating the env and install the requirements.
  ```bash
  git clone https://github.com/huyluongme/TMA_AI_Advance_Group_51.git

  cd TMA_AI_Advance_Group_51 

  conda create -n tma_ai python=3.10 -y

  conda activate tma_ai

  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

  pip install -r requirements.txt

  ```
## 4. Download Dataset & Data Processing.
  ```bash
  cd data_preprocess
  git clone https://github.com/spMohanty/PlantVillage-Dataset.git
  python split_dataset.py
  python augment.py

  ```
## 5. Train Model.
  ```bash
  python train_test_resnet18.py
  
  ```
