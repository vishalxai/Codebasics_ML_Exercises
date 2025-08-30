# Patient Health Risk Prediction – Linear, Lasso & Ridge Regression

## 📌 Project Overview
This project uses a healthcare dataset (`patient_health_data.csv`) containing patient health indicators such as **age, BMI, blood pressure, cholesterol, glucose, insulin, heart rate, activity level, diet quality, smoking status, alcohol intake**, and a **health risk score**.  
The goal is to build regression models that predict the **health risk score** and evaluate the impact of **regularization techniques**.

## ⚙️ Tasks Performed
1. **Data Preparation**
   - Imported dataset & performed sanity checks.
   - Handled categorical encoding (`smoking_status` → 0/1).
   - Checked for missing values.

2. **Model Training**
   - Trained **Linear Regression** as a baseline.
   - Applied **Lasso Regression (L1)** with different alpha values.
   - Applied **Ridge Regression (L2)** with different alpha values.

3. **Evaluation**
   - Used **R² score** to measure model performance.
   - Compared Linear vs. Lasso vs. Ridge results.

## 🔑 Key Learnings
- **Linear Regression** provides the baseline without regularization.  
- **Lasso (L1)** helps with **feature selection** by setting some coefficients to zero → useful when dataset has irrelevant features.  
- **Ridge (L2)** shrinks coefficients but keeps all features → better for handling **multicollinearity**.  
- **Tuning alpha** is crucial: too low = overfitting risk, too high = underfitting.

## 🛠️ Tech Stack
- Python, Pandas, NumPy  
- Scikit-learn (LinearRegression, Lasso, Ridge)  
- Matplotlib for visualization  

---
📊 This exercise demonstrates how **regularization improves generalization** and prevents overfitting in regression models.