# Cancer Prediction using Decision Tree and Random Forest

## 📌 Project Overview
This project applies **Decision Tree** and **Random Forest** models to predict cancer diagnosis based on patients’ medical and lifestyle features. The dataset includes 1500 patients with features such as age, BMI, smoking habits, genetic risk, physical activity, and alcohol intake.

The goal is to:
1. Explore and preprocess the dataset.
2. Train and evaluate a **Decision Tree Classifier**.
3. Train and evaluate a **Random Forest Classifier**.
4. Tune Random Forest hyperparameters to improve performance.
5. Compare models and interpret their results in the context of medical decision-making.

---

## 📊 Dataset
- **Source:** [Kaggle Cancer Prediction Dataset](https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset)  
- **Rows:** 1500 patients  
- **Target:** `diagnosis` (0 = No Cancer, 1 = Cancer)  

**Features:**
- `age`: Patient’s age (20–80)  
- `gender`: 0 = Male, 1 = Female  
- `bmi`: Body Mass Index (15–40)  
- `smoking`: 0 = No, 1 = Yes  
- `genetic_risk`: 0 = Low, 1 = Medium, 2 = High  
- `physical_activity`: Hours/week (0–10)  
- `alcohol_intake`: Drinks/week (0–5)  
- `cancer_history`: 0 = No, 1 = Yes  
- `diagnosis`: Target (0 = No Cancer, 1 = Cancer)  

---

## ⚙️ Methods

### 🔹 Task 1: Data Preparation
- Loaded dataset (`cancer_data.csv`)
- Checked for missing values (none found)
- Explored distributions and class balance:
  - No Cancer: 943 (62.8%)
  - Cancer: 557 (37.1%)

### 🔹 Task 2: Decision Tree Classifier
- Accuracy: **84.5%**
- Cancer recall: **0.81**
- Observed **overfitting risk** with a single tree.

### 🔹 Task 3: Random Forest Classifier
- Accuracy: **92%**
- Cancer recall: **0.86**
- Improved stability by averaging 25 trees.

### 🔹 Task 4: Tuned Random Forest
Parameters:
- `n_estimators=50`
- `max_features="log2"`
- `criterion="entropy"`
- `bootstrap=False`
- `max_depth=15`
- `min_samples_split=5`
- `min_samples_leaf=3`

**Results:**
- Accuracy: **94.4%**
- Cancer recall: **0.89**
- Fewer false negatives, better suited for medical prediction.

---

## 📈 Results Comparison

| Model                 | Accuracy | Recall (Cancer) | Notes |
|------------------------|----------|-----------------|-------|
| Decision Tree          | 84.5%    | 0.81            | High variance |
| Random Forest (25 est) | 92.0%    | 0.86            | Balanced |
| Tuned Random Forest    | 94.4%    | 0.89            | Best model |

---

## 🔍 Key Learnings
- **Ensembles > Single Tree**: Random Forest reduced overfitting and improved generalization.
- **Hyperparameter tuning**: Adjusting depth, splits, and number of estimators significantly boosted performance.
- **Medical ML Insight**: Recall (sensitivity) is more important than raw accuracy, since false negatives in cancer prediction can be dangerous.
- **Practical deployment**: Would further optimize thresholds using ROC–AUC to catch more positive cases.

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone <repo-link>

pip install pandas numpy matplotlib seaborn scikit-learn

jupyter notebook