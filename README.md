
# Cameroon-pidgin-news-classification

# CamPidNEWS: News Topic Classification for Cameroonian Pidgin English

This repository contains the dataset, code, and results for our news topic
classification project on Cameroonian Pidgin English, developed as part of
the NLP and Large Language Models course at AIMS Cameroon (March 2026).

---

## Overview

We construct **CamPidNEWS**, the first news topic classification dataset for
Cameroonian Pidgin English — a language not covered in any existing NLP
benchmark. We evaluate classical machine learning baselines and a fine-tuned
multilingual transformer model on this dataset.

---

## Language

**Cameroonian Pidgin English** is an English-based creole widely spoken across
Cameroon as a lingua franca. It is related to Nigerian Pidgin (Naija) but
contains Cameroon-specific vocabulary and references. Key linguistic features
include:

- `don` — perfective aspect marker (has/have done)
- `dem` — plural marker
- `dis / dat` — this / that
- `kontri` — country

---

## Dataset

The dataset consists of **1,150 news headlines** in Cameroonian Pidgin English,
generated using Large Language Models with Cameroon-specific prompts and
manually verified for linguistic authenticity.

### Topic Categories

| Label         | Description                        |
|---------------|------------------------------------|
| Politics      | Government, elections, policy      |
| Sports        | Football, athletics, competitions  |
| Business      | Economy, trade, finance            |
| Technology    | Digital, innovation, ICT           |
| Health        | Medicine, public health, hospitals |
| Entertainment | Music, arts, culture, events       |

### Data Splits

| Split       | Size  | Percentage |
|-------------|-------|------------|
| Train       | 900   | 78.3%      |
| Development | 100   | 8.7%       |
| Test        | 150   | 13.0%      |
| **Total**   | **1,150** | **100%** |

### Train Label Distribution

| Label         | Count |
|---------------|-------|
| Politics      | 163   |
| Business      | 151   |
| Sports        | 149   |
| Health        | 149   |
| Technology    | 149   |
| Entertainment | 139   |

### Dataset Format

Each file is a CSV with the following columns:

| Column    | Description                              |
|-----------|------------------------------------------|
| `id`      | Unique identifier for each headline      |
| `headline`| News headline in Cameroonian Pidgin English |
| `label`   | Topic category (one of the six labels)   |
| `split`   | Dataset split (train / test)             |

### Sample Headlines

| Headline                                                        | Label         |
|-----------------------------------------------------------------|---------------|
| Ministry of interior don announce new identification card       | politics      |
| PWD Bamenda don win Cameroon Cup final for Yaounde              | sports        |
| Ecobank don expand service to all ten region for Cameroon       | business      |
| Kontri education ministry don introduce coding in curriculum    | technology    |
| Health ministry don distribute medicine to rural clinic dem     | health        |
| Douala carnival don record biggest attendance for dis year      | entertainment |

---

## Repository Structure

```
CamPidNEWS/
├── data/
│   ├── train.csv           # 1,000 training headlines (before dev split)
│   └── test.csv            # 150 test headlines
├── results/
│   ├── model_comparison.png    # Bar chart: all model F1 scores
│   ├── training_curves.png     # Loss and F1 curves over 10 epochs
│   ├── confusion_matrix.png    # Confusion matrix for AfroXLMR-base
│   ├── per_class_f1.png        # Per-class F1 for AfroXLMR-base
│   └── test_predictions.csv    # All 150 test predictions with labels
├── NLP_code.ipynb          # Full experiment notebook
└── README.md
```

---

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
pip install transformers datasets torch
```

The fine-tuning experiments were run on **Google Colab with a T4 GPU**.

---

## Reproduction Instructions

### Step 1 — Mount Drive and Load Data

```python
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
train_df = pd.read_csv('/content/drive/MyDrive/nlp_data/train.csv')
test_df  = pd.read_csv('/content/drive/MyDrive/nlp_data/test.csv')
```

### Step 2 — Create Dev Split

```python
from sklearn.model_selection import train_test_split

train_final, dev_df = train_test_split(
    train_df, test_size=0.10, random_state=42,
    stratify=train_df['label']
)
train_final.to_csv('train_split.csv', index=False)
dev_df.to_csv('dev_split.csv', index=False)
```

### Step 3 — Run Classical ML Baselines

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
import xgboost as xgb

vectorizer  = CountVectorizer()
X_train_vec = vectorizer.fit_transform(train_final['headline'])
X_test_vec  = vectorizer.transform(test_df['headline'])

le          = LabelEncoder()
y_train_enc = le.fit_transform(train_final['label'])
y_test_enc  = le.transform(test_df['label'])

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_vec, y_train_enc)

# MLP
mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=50, random_state=42)
mlp.fit(X_train_vec, y_train_enc)

# XGBoost
xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=6,
                              random_state=42, eval_metric='mlogloss')
xgb_clf.fit(X_train_vec, y_train_enc)
```

### Step 4 — Fine-tune AfroXLMR-base

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import torch

MODEL_NAME = "Davlan/afro-xlmr-base"
LABEL_LIST = ['business', 'entertainment', 'health', 'politics', 'sports', 'technology']
label2id   = {l: i for i, l in enumerate(LABEL_LIST)}
id2label   = {i: l for l, i in label2id.items()}

training_args = TrainingArguments(
    output_dir       = './results',
    num_train_epochs = 10,
    per_device_train_batch_size = 64,
    per_device_eval_batch_size  = 64,
    learning_rate    = 2e-5,
    eval_strategy    = 'epoch',
    save_strategy    = 'epoch',
    load_best_model_at_end  = True,
    metric_for_best_model   = 'weighted_f1',
    logging_steps    = 10,
    seed             = 42,
)
```

Open `NLP_code.ipynb` for the full end-to-end code including tokenization,
training, evaluation, and all visualizations.

---

## Results

### Model Comparison (Weighted F1 on Test Set, n=150)

| Model          | Type          | Weighted F1 |
|----------------|---------------|-------------|
| Naive Bayes    | Classical ML  | 83.34%      |
| MLP            | Classical ML  | 87.33%      |
| XGBoost        | Classical ML  | 79.22%      |
| **AfroXLMR-base** | **Fine-tuned LM** | **95.98%** |

### AfroXLMR-base Per-Class Results

| Label         | Precision | Recall | F1     | Support |
|---------------|-----------|--------|--------|---------|
| Business      | 0.9615    | 0.8929 | 0.9259 | 28      |
| Entertainment | 0.9643    | 0.9643 | 0.9643 | 28      |
| Health        | 1.0000    | 1.0000 | 1.0000 | 25      |
| Politics      | 0.9231    | 0.9600 | 0.9412 | 25      |
| Sports        | 0.9615    | 0.9615 | 0.9615 | 26      |
| Technology    | 0.9474    | 1.0000 | 0.9730 | 18      |
| **Weighted avg** | **0.9604** | **0.9600** | **0.9598** | **150** |

### Overall Test Statistics

| Metric              | Value  |
|---------------------|--------|
| Total test examples | 150    |
| Correct predictions | 144    |
| Wrong predictions   | 6      |
| Accuracy            | 96.00% |

### AfroXLMR-base Training (Validation F1 per Epoch)

| Epoch | Validation F1 |
|-------|---------------|
| 1     | 47.98%        |
| 2     | 76.19%        |
| 3     | 78.01%        |
| 4     | 86.48%        |
| 5     | 89.90%        |
| 6     | 91.86%        |
| **7** | **91.91%** (best) |
| 8     | 91.90%        |
| 9     | 90.94%        |
| 10    | 90.94%        |

---

## Error Analysis

AfroXLMR-base made only **6 errors** out of 150 test examples:

| ID  | True Label    | Predicted Label |
|-----|---------------|-----------------|
| 19  | Business      | Politics        |
| 85  | Entertainment | Sports          |
| 89  | Business      | Politics        |
| 99  | Politics      | Business        |
| 136 | Business      | Technology      |
| 150 | Sports        | Entertainment   |

The most frequent confusion is between **Business and Politics**, which is
expected as government economic policy headlines overlap both categories
in Cameroonian Pidgin news.

---

## Acknowledgements

This work is inspired by the MasakhaNEWS benchmark (Adelani et al., 2023).
We thank AIMS Cameroon for providing the computational resources and
course structure that made this project possible.
