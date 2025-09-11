# Breast Cancer Recurrence Classification  
Machine Learning project to predict whether a patient will experience breast cancer recurrence based on clinical features.  

---

## 📌 Problem Statement  
Breast cancer recurrence prediction is a critical task in oncology.  
The goal of this project is to build and evaluate machine learning models that can classify patients into:  
- **0 → No recurrence**  
- **1 → Recurrence events**  

Handling **class imbalance** is the major challenge in this dataset.  

---

## 📊 Dataset  
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/breast+cancer)  
- **Size:** 286 entries, 9 categorical + numerical features.  
- **Target variable:** `class` → {`no-recurrence-events`, `recurrence-events`}  

Key features include:  
- Age, Menopause status, Tumor size, Lymph node involvement (`inv-nodes`), Node-caps, Breast side, Breast quadrant, Irradiation.  

---

## ⚙️ Methodology  

### 🔹 Step 1: Data Preparation  
- Converted categorical features using **one-hot encoding**.  
- Converted age, tumor-size, inv-nodes into **numeric bins**.  
- Handled missing values using **median imputation**.  

### 🔹 Step 2: Baseline Model (Logistic Regression)  
- Split dataset into **train/test (80/20)** with stratification.  
- Trained a **logistic regression** classifier.  
- Evaluated with **classification report, confusion matrix, ROC-AUC**.  

### 🔹 Step 3: Handling Class Imbalance  
Tried multiple resampling techniques:  
1. **Baseline (no resampling)**  
2. **Random Undersampling**  
3. **SMOTE (Synthetic Minority Oversampling)**  
4. **SMOTE + Tomek Links**  

---

## 📈 Results  

### 🔹 Classification Reports  

| Method                 | Precision (Recurrence) | Recall (Recurrence) | F1-Score (Recurrence) | Accuracy |
|-------------------------|-------------------------|----------------------|------------------------|----------|
| Baseline (Imbalanced)  | 0.4286                  | 0.1765               | 0.2500                 | 68.9%    |
| Random Undersampling   | 0.3448                  | 0.5882               | 0.4348                 | 55.1%    |
| SMOTE Oversampling     | 0.4500                  | 0.5294               | 0.4865                 | 67.2%    |
| SMOTE + Tomek Links    | 0.4500                  | 0.5294               | 0.4865                 | 67.2%    |

---

### 🔹 Confusion Matrices  

#### Baseline  
High accuracy, but **very poor recall** for recurrence cases (model ignores minority class).  

#### Random Undersampling  
Improves recall but hurts overall accuracy (data loss).  

#### SMOTE / SMOTE + Tomek Links  
Balanced results with improved recall and F1-score for recurrence events.  

---

## 📊 Visualizations  

### Confusion Matrix (SMOTE)  
![Confusion Matrix SMOTE](images/confusion_matrix_smot.png)

### Comparison of Methods on Minority Class Recall  
![Comparison of Methods](images/comparison_of_methods_on_minorityclass.png)

---

## 🚀 Key Learnings  
- Accuracy is misleading in imbalanced datasets.  
- **Recall for recurrence events is critical** (missing a recurrence is riskier than a false alarm).  
- SMOTE-based oversampling gave the **best balance** between recall and overall performance.  

---

## 🛠️ Tech Stack  
- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Imbalanced-learn, Matplotlib, Seaborn, Jupyter  

---

## 📌 Next Steps  
- Try **ensemble models** (Random Forest, XGBoost) with imbalance handling.  
- Apply **hyperparameter tuning** (GridSearchCV).  
- Deploy model via **Streamlit or Flask** for real-time prediction.  

---

## 👨‍💻 Author  
**Vishal Singh**  
- AI/ML Engineer | Focused on Machine Learning, Data Science, and Deployment.  
- [LinkedIn](www.linkedin.com/in/vishalxai) | [GitHub](https://github.com/vishalxai)  