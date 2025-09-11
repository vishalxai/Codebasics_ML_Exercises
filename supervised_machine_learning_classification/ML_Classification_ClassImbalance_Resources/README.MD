# Handling Class Imbalance — Customer Churn (Codebasics exercise)

## Project overview
This repository contains a small end-to-end exercise for handling class imbalance on a customer churn dataset (UCI / Codebasics-style notebook).  
The goal is to explore how resampling techniques (undersampling, SMOTE, SMOTE + Tomek links) affect classifier performance — especially **recall** on the minority class.

Key experiments included:
- Baseline logistic regression
- Random undersampling
- SMOTE (Synthetic Minority Over-sampling Technique)
- SMOTE + Tomek links (combine oversampling + cleaning)
- Model evaluation using classification report and confusion matrix

## Files added
- `data/churn.csv` — dataset used for the exercise (UCI churn-like data).
- `notebooks/Handling_classimbalance_customerchurn.ipynb` — main notebook with exploration, baseline model, resampling experiments and plots.
- `notebooks/class_imbalance.ipynb` — supporting notebook / notes (alternate steps and debugging).

## Requirements
Create a Python environment and install dependencies. Recommended Python >= 3.9.

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

pip install --upgrade pip
pip install -r requirements.txt