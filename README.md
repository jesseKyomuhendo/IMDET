# IMDET - Image Manipulation Detection
### Repository GitHUB Link: https://github.com/jesseKyomuhendo/IMDET

A two-stream hybrid CNN for detecting and classifying image manipulations.
Detects two types: **copy-move** and **splicing**, plus authentic images.

---

## Project Structure

```
IMDET/
├── data/                  - Dataset folder (not included in repo, download separately)
├── model/                 - Trained model (included in submission ZIP)
├── results/               - Evaluation results saved here (JSON)
├── settings/
│   └── SettingsAssistant.py
├── source/
│   ├── dataset.py         - Data pipeline. Loads images from disk in batches and prepares them for the model (resize, normalize, augment)
│   ├── evaluate.py        - Evaluation functions
│   ├── model.py           - Two-stream architecture
│   ├── split_dataset.py   - Dataset split script
│   └── train.py           - Training script
├── config.yaml            - All parameters and paths
├── requirements.txt
├── README.md
└── test.py                - Run this to reproduce results
```

---

## Setup (Not needed if you are testing on Google Colab)

> **Windows users:** TensorFlow requires Python 3.11 or lower.

### 1. Create a Virtual Environment

**Windows:**
```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset

The dataset is not included in the repository due to size.

### Datasets and Splits
Download from: https://drive.google.com/drive/folders/1xT-BTCMnHEe1R45objlO6cIzZlpsdOrE?usp=sharing

- To download click the `Download all`  button in top right corner. (This will let you download the folder as one zip file)
- The download zip file should be renamed to:`Data.zip`

The ZIP contains:
- CASIA2.0 image folders (authentic, copy_move, splicing)
- Dataset splits (train.csv, val.csv, test.csv)

---

## Testing the Model on Google Colab

> **Note:** Running test.py on native Windows may crash due to a known
> TensorFlow limitation with large models on Windows >= 2.11.
> We recommend using Google Colab to test the model.

### Step 1 — Open a new notebook
Go to Google Colab and open a new notebook.
Open the Terminal from the left sidebar (the `>_` icon).

### Step 2 — Upload the submission ZIP
Upload `Project-Group05.zip` via the Files panel (upload button in the left sidebar).
Wait for the upload to complete before proceeding.

### Step 3 — Unzip the source code 
```bash
unzip /content/Project-Group05.zip -d /content/
cd /content/IMDET
```

### Step 4 — Upload and unzip the dataset
Upload `Data.zip` via the Files panel, then extract:
```bash
unzip /content/Data.zip -d /content/
```

### Step 5 — Place `CASIA2.0` & `splits` folders into `data` folder of the repository
- After unzipping the `Data.zip` file you will get a folder called `Data`
- Inside `Data` are 2 folder called `CASIA2.0` & `splits`
- Move `CASIA2.0` & `splits` into `data` folder of the repository/source code 

### Step 6 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 7 — Run test
```bash
python test.py
```

Results will be printed to the terminal and saved to `results/`.

---

## Training

Training was conducted on Google Colab Pro using an NVIDIA A100 GPU.
To retrain the model from scratch:

```bash
python source/train.py
```

> **Note:** Training on CPU is very slow (~30-60 min/epoch).
> We recommend Google Colab for training.

The imbalance strategy and all other hyperparameters can be configured in `config.yaml`.

---

## Configuration

All parameters are defined in `config.yaml`. No hardcoded values exist in any source file.

Key settings:
```yaml
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.00001
  early_stopping_patience: 45
  imbalance_strategy: class_weights

model:
  backbone: resnet50
  pretrained: true
  num_classes: 3
```


---

## Group 5
- Sambrina Selvarajah
- Sandra Østrem
- Jesse Kyomuhendo Tibamwenda