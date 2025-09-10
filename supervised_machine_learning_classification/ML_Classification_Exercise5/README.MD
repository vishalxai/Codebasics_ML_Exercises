# 🐾 Pet Adoption Prediction using Decision Trees

This project predicts the **likelihood of pet adoption** using supervised machine learning.  
It is part of the Codebasics ML Exercises and demonstrates data preprocessing, decision tree modeling, and evaluation techniques.

---

## 📌 Problem Statement
Given a dataset (`pet_adoption_data.csv`) with features such as:
- `pet_type`, `breed`, `age_months`, `color`, `size`, `weight_kg`
- `vaccinated`, `health_condition`, `timein_shelter_days`, `adoption_fee`
- `previous_owner`
- **Target**: `adoption_likelihood` (0 = unlikely, 1 = likely)

The task is to build a **Decision Tree Classifier** that predicts whether a pet is likely to be adopted.

---

## ⚙️ Steps Performed
1. **Data Preparation & Exploration**
   - Loaded dataset, checked shape (2007 rows, 13 columns)
   - Dropped non-informative columns (`pet_id`)
   - Visualized distributions of target and numerical features

2. **Data Encoding & Scaling**
   - Encoded categorical variables:
     - `size`: mapped (Small=1, Medium=2, Large=3)
     - `color`, `pet_type`, `breed`: one-hot encoding
   - Scaled numerical features:
     - `weight_kg`: MinMaxScaler
     - `adoption_fee`: StandardScaler

3. **Model Training**
   - Used **DecisionTreeClassifier**
   - Baseline model with default settings
   - Tuned hyperparameters (`criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf`)

4. **Evaluation**
   - Accuracy on train/test sets
   - Classification report (precision, recall, f1-score)
   - Confusion matrix (visualized with heatmap)

---

## 📊 Results
- **Baseline Model**
  - Train Accuracy: 100% (overfit)
  - Test Accuracy: ~87%
- **After Hyperparameter Tuning**
  - Balanced performance with reduced overfitting
  - Good precision & recall for both classes

**Confusion Matrix Example:**
[[367  38]
[ 39 159]]

- Class `0` (Not Adopted): 91% recall
- Class `1` (Adopted): 80% recall

---

## 📈 Key Learnings
- Decision Trees can easily overfit; hyperparameters like `max_depth`, `min_samples_leaf`, and `min_samples_split` are crucial for regularization.
- Accuracy alone is misleading; precision, recall, and confusion matrices give deeper insight.
- Data preprocessing (encoding + scaling) is critical when mixing categorical and numerical features.

---

## 🔮 Next Steps
- Try **Random Forests** or **Gradient Boosted Trees** for improved performance
- Perform **cross-validation** for robust model selection
- Explore **class weights** if dataset imbalance is significant

---

## 📂 Files
- `assignment.ipynb` → Your working notebook
- `assignment_solution.ipynb` → Final solved notebook
- `pet_adoption_data.csv` → Dataset
- `README.md` → Project documentation (this file)

---

## 👨‍💻 Author
Vishal — AI/ML Engineer
Building end-to-end ML projects 