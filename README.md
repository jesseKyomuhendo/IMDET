# IMDET - Image Manipulation Detection

A two-stream hybrid CNN for detecting and classifying image manipulations.
Detects two types: **copy-move** and **splicing**, plus authentic images.

---

## Project Structure

```
IMDET/
├── data/
│   ├── CASIA2.0/          ← Primary training dataset (download separately)
│   ├── Columbia/          ← Cross-dataset evaluation (download separately)
│   └── splits/            ← Auto-generated CSV splits (train/val/test)
├── model/                 ← Trained model saved here after training
├── results/               ← Evaluation results saved here (JSON)
├── settings/
│   └── SettingsAssistant.py
├── source/
│   ├── dataset.py         ← Data pipeline
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

## Dataset Setup

Datasets are not included in the repository due to size. Download and place them manually.

### CASIA 2.0 (Primary — ~2.6 GB)


### Columbia Uncompressed (Cross-dataset evaluation — ~400 MB)

---

## Running the Project

### Step 1 — Generate dataset splits
```bash
python source/split_dataset.py
```
Creates `data/splits/train.csv`, `val.csv`, `test.csv`.

### Step 2 — Train the model
```bash
python source/train.py
```
Trains the model and saves the best version to `model/best_model.keras`.

> **Note:** Training on CPU is very slow (~30-60 min/epoch).
> We use Google Colab 

### Step 3 — Evaluate (reproduce results)
```bash
python test.py
```
Loads the trained model and runs evaluation on the test set.
Prints accuracy, F1, AUC-ROC, and confusion matrix.
Saves results to `results/`.

---

## Configuration

All parameters are defined in `config.yaml`. No hardcoded values exist in any source file.



---
