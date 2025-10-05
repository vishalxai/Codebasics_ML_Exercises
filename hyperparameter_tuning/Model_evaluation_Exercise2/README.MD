# 🏨 Hotel Reservations Classification — Machine Learning Project

This project is part of the **Codebasics Machine Learning Course** and focuses on predicting **whether a hotel reservation will be canceled or not**.  
The dataset contains detailed booking information such as the number of adults, children, booking lead time, price per room, and more.

---

## 📂 Dataset
**Dataset Name:** `hotel_reservations.csv`  
**Source:** [Kaggle - Hotel Reservations Classification Dataset](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset)

### Features
- `no_of_adults`: Number of adults  
- `no_of_children`: Number of children  
- `lead_time`: Days between booking and arrival  
- `avg_price_per_room`: Average price per day  
- `booking_status`: Target variable (Canceled / Not Canceled)  
…and more booking-related details like meal plan, room type, etc.

---

## 🧹 Task 1: Data Preprocessing and EDA
- Imported the dataset and performed an overview check (`df.head()`, `df.shape`)
- Dropped irrelevant columns (`booking_id`, `arrival_year`, `arrival_month`, `arrival_date`)
- Visualized data:
  - **Booking status countplot**
  - **Boxplot for lead time and price per room**
- Encoded categorical variables using **One-Hot Encoding**

---

## 📏 Task 2: Feature Scaling
- Scaled numerical features (`lead_time`, `avg_price_per_room`) using `StandardScaler`
- Defined:
  - **X** = Features
  - **y** = Target variable (`booking_status`)

---

## 🔁 Task 3: Implementing K-Fold Cross Validation
- Used **KFold(n_splits=7, shuffle=True)**  
- Compared models:
  - **Naive Bayes**
  - **Logistic Regression**
  - **Decision Tree**
- Evaluated each model’s **average cross-validation accuracy**

---

## ⚙️ Task 4: Stratified K-Fold Cross Validation
- Ensured class balance using `StratifiedKFold`
- Compared results to regular K-Fold for consistency

---

## 🔍 Task 5: Grid Search CV
- Applied **GridSearchCV** to tune:
  - **Naive Bayes:** `var_smoothing` = [1e-9, 1e-8, 1e-7]
  - **Random Forest:** `n_estimators` = [10, 50, 100], `max_depth` = [5, 10, 15]
- Extracted **best parameters** for each model

---

## 🎲 Task 6: Randomized Search CV
- Used **RandomizedSearchCV** for `RandomForestClassifier`
- Parameters:
  - `n_estimators`: [10, 50, 100]
  - `max_depth`: [5, 10, 15, 20]
  - `cv = 7`, `n_iter = 3`
- Printed **best parameters** and performance summary

---

## 📊 Results Summary
| Model | Method | Accuracy (approx) |
|-------|---------|------------------|
| Naive Bayes | K-Fold CV | 0.40 |
| Logistic Regression | K-Fold CV | 0.80 |
| Decision Tree | K-Fold CV | 0.85 |
| Random Forest (Tuned) | Randomized Search CV | 0.88 |

---

## 🧩 Key Learnings
- Practiced **cross-validation** and **model selection techniques**
- Understood **hyperparameter tuning** using GridSearchCV & RandomizedSearchCV
- Observed how **Random Forest** performance improves with tuning
- Reinforced workflow: *EDA → Feature Scaling → Model Evaluation → Hyperparameter Optimization*

---

## 🧑‍💻 Tech Stack
- **Python**
- **Scikit-learn**
- **Matplotlib**, **Seaborn**
- **Pandas**, **NumPy**
- **XGBoost**

---

## 📘 Author
**Vishal** — AI/ML Engineer
Real-world ML projects as part of the Codebasics ML course.
