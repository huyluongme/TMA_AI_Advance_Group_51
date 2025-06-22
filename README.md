# TMA AI Advance Group 51 - Plant Disease Detection
## 1. Demo
In Progess...
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
