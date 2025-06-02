## **Data Understanding**

- **Total Images**: Approximately 54305 RGB images.
- **Image Resolution**: Standardized at 256×256 pixels.
- Images are categorized into 38 classes (folder), combining plant species and disease types.
- Each folder is labeled in the format `PlantName___DiseaseName` (e.g., `Apple___Apple_scab`).
- Covers 14 crop species: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.
- Images are standardized at a resolution of 256×256 pixels.

## **Data Preparation**

- Split train and test data set
    - Train set 80%
    - Test set 20%
- Apply data augmentation (rotation, flip, zoom) to increase model generalization

### Set up environment

- Download Python 3.10 directly from the official site: https://www.python.org/downloads/release/python-3100/
    - Choose the **Windows installer (64-bit)** version, and remember to:
    **Check “Add Python to PATH”** during installation.
- Download Git for Windows from: https://git-scm.com/downloads
    - Install with default options.
- Create a Virtual Environment (Recommended)
    
    ```bash
    mkdir plant_disease_project
    cd plant_disease_project
    
    # Create a virtual environment
    python -m venv venv
    venv\Scripts\activate
    
    # Clone the PlantVillage Dataset
    git clone https://github.com/spMohanty/PlantVillage-Dataset.git
    ```
    
- Install Install Microsoft C++ Redistributables
    - Go here and **install x64 versions:** https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist
    - Look for **Visual Studio 2015–2022** Redistributables (`vc_redist.x64.exe` )
- Install packets

```bash
pip install --upgrade pip
pip install tensorflow==2.15.0 matplotlib
pip install scipy
pip install tqdm
```

### Split dataset

Create file split_dataset.py at plant_disease_project folder

Execute python script:

```bash
python split_dataset.py
```

TRAIN_RATIO = 0.8 # ratio now is 80%, can adjust 70% or 75%

Folder PlantVillage-Dataset\PlantVillage-Dataset\raw\split should be created which containing train and test folder.

- 80% images of each class is copied to train folder
- 20% images of each class is copied to test folder

### Apply data augmentation on train set

- Using pytorch + CUDA GPU

Create file augment.py at plant_disease_project folder

Execute python script:

```bash
python augment.py
```
Wait for about 10 min


Total images = 54305

Train set = (total images) x 80% = ~43529

After apply data augmentation on train set

AUGMENTATIONS_PER_IMAGE = 3 mean each original images in train set ⇒ generate 3 new images

Then 1 original + 3 new images = 4

Total data in train dataset = 43529 x 4 = 173716

Images name file has prefix aug_<number>_… are augmented