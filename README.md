# News Topic Classification in Cameroonian Pidgin English

## Overview
This project investigates the effectiveness of classical machine learning models and a fine-tuned multilingual transformer model for news topic classification in **Cameroonian Pidgin English**.

The work was conducted as part of the **Natural Language Processing and Large Language Models course at the African Institute for Mathematical Sciences (AIMS Cameroon)**.

The objective is to evaluate whether multilingual transformer models such as **AfroXLMR** can achieve strong performance on a low-resource African language classification task.

---

## Research Objective
The study aims to:

- Construct a dataset of Cameroonian Pidgin English news headlines.
- Train classical machine learning baseline models.
- Fine-tune the **AfroXLMR multilingual transformer model** for topic classification.
- Compare the performance of classical and transformer-based approaches.

---

## Dataset

### Data Source
The dataset consists of news headlines generated in **Cameroonian Pidgin English**.

The headlines were generated using **Claude AI** based on prompts designed to produce examples corresponding to six news categories. The generated headlines were subsequently **reviewed and corrected by a fluent speaker of Cameroonian Pidgin English** to ensure linguistic accuracy.

### Topic Categories
The dataset contains six topic labels:

- Business
- Entertainment
- Health
- Politics
- Sports
- Technology

### Dataset Distribution (Training Set)

| Category | Samples |
|--------|--------|
| Politics | 163 |
| Business | 151 |
| Sports | 149 |
| Health | 149 |
| Technology | 149 |
| Entertainment | 139 |

### Dataset Split

| Dataset | Samples |
|-------|-------|
| Training | 900 |
| Development | 100 |
| Test | 150 |

---

## Methods

### Classical Machine Learning Models
The following baseline models were trained using a **bag-of-words representation** produced by `CountVectorizer`:

- Multinomial Naive Bayes
- Multi-Layer Perceptron (MLP)
- XGBoost

### Transformer Model
A multilingual transformer model was used:

- **AfroXLMR-base**
- Fine-tuned for the news topic classification task

---

## Results

### Model Performance (Test Set)

| Model | Weighted F1 Score (%) |
|------|-----------------------|
| Naive Bayes | 83.34 |
| MLP | 87.33 |
| XGBoost | 79.22 |
| AfroXLMR (Fine-tuned) | **95.98** |

The fine-tuned AfroXLMR model achieved the highest performance across all evaluated models.

---

## Repository Structure
