# Titanic Survival Prediction 🚢

This project predicts whether a Titanic passenger survived or not using the Titanic dataset.  
It applies **Gaussian Naive Bayes** as a classification model and evaluates its performance.

---

## 📌 Problem Statement
Predict passenger survival (`0 = Died`, `1 = Survived`) using features such as class, sex, age, and fare.  
Dataset: `titanic.csv`

---

## 📊 Dataset Features
- **p_class**: Passenger class (1st, 2nd, 3rd)  
- **sex**: Gender of the passenger  
- **age**: Age of the passenger  
- **fare**: Ticket fare paid  
- **survived**: Target variable (0 = Died, 1 = Survived)  

Dropped: `passenger_id`, `name`, `ticket`, `cabin` (low predictive value).

---

## 🔎 Steps Performed
1. **Data Preparation & Exploration**
   - Loaded and inspected dataset.
   - Checked for missing values.
   - Visualized distributions: survival counts, passenger class, sex, age, fare.

2. **Data Preprocessing**
   - Filled missing `age` and `fare` values with medians.
   - Encoded `sex` using one-hot encoding.
   - Standardized `fare` using StandardScaler.
   - Selected final features: `p_class`, `sex`, `age`, `fare`.

3. **Model Training**
   - Split data into 70% training and 30% testing.
   - Trained Gaussian Naive Bayes classifier.

4. **Model Evaluation**
   - Accuracy: **76%**
   - Classification Report (Precision, Recall, F1-scores).
   - Confusion Matrix for visual evaluation.

---

## 📈 Results
- **Accuracy**: 76%  
- Confusion Matrix:  

|               | Predicted Died | Predicted Survived |
|---------------|----------------|---------------------|
| **Actual Died**     | 128            | 37                  |
| **Actual Survived** | 27             | 76                  |

- Model predicts **deaths better than survivals**, likely due to class imbalance.

---

## 🚀 Future Improvements
- Try Logistic Regression, Random Forest, or XGBoost.
- Handle class imbalance (SMOTE, class weights).
- Add feature engineering (family size, title extraction from names).

---

## 🛠️ Tech Stack
- Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)  
- Jupyter Notebook  

---

## 📂 Repository Structure
├── assignment.ipynb            # Notebook with step-by-step solution
├── assignment_solution.ipynb   # Final solution notebook
├── titanic.csv                 # Dataset
└── README.md                   # Project documentation

---

✍️ Author: Vishal