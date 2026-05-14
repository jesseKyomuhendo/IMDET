# IMDET - Image Manipulation Detection

A two-stream hybrid CNN for detecting and classifying image manipulations.
Detects two types: **copy-move** and **splicing**, plus authentic images.

---

## Project Structure

```
IMDET/
├── data/                  ← Dataset folder (not included in repo, download separately)
├── model/                 ← Trained model (included in submission ZIP)
├── results/               ← Evaluation results saved here (JSON)
├── settings/
│   └── SettingsAssistant.py
├── source/
│   ├── dataset.py         ← Data pipeline. Loads images from disk in batches and prepares them for the model (resize, normalize, augment)
│   ├── evaluate.py        ← Evaluation functions
│   ├── model.py           ← Two-stream architecture
│   ├── split_dataset.py   ← Dataset split script
│   └── train.py           ← Training script
├── config.yaml            ← All parameters and paths
├── requirements.txt
├── README.md
└── test.py                ← Run this to reproduce results
```

---

## Setup

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

### CASIA2.0 ZIP
Download from: [CASIA2.0 DOWNLOAD LINK — TO BE PROVIDED]

The ZIP contains:
- Organised image folders (authentic, copy_move, splicing)
- Dataset splits (train.csv, val.csv, test.csv)

---

## Reproducing Results on Google Colab

> **Note:** Running test.py on native Windows may crash due to a known
> TensorFlow limitation with large models on Windows >= 2.11.
> We recommend using Google Colab to reproduce results.

### Step 1 — Open a new notebook
Go to https://colab.research.google.com and open a new notebook.
Open the Terminal from the left sidebar (the `>_` icon).

### Step 2 — Upload the submission ZIP
Upload `Project-Group05.zip` via the Files panel (upload button in the left sidebar).
Wait for the upload to complete before proceeding.

### Step 3 — Unzip the submission
```bash
unzip /content/Project-Group05.zip -d /content/IMDET
cd /content/IMDET
```

### Step 4 — Upload and unzip the dataset
Upload `CASIA2.0.zip` via the Files panel, then extract:
```bash
unzip /content/CASIA2.0.zip -d /content/IMDET/data
```

### Step 5 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 6 — Run test
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

## AI Tools

This project used Claude (Anthropic) as a development assistant for code scaffolding and debugging.
All code was reviewed, tested, and adapted by the group members.

---

## Group 5
- Sambrina Selvarajah
- Sandra Østrem
- Jesse Kyomuhendo Tibamwenda
