# 🍄 Mushroom Classification using Ensemble Learning

This project focuses on predicting whether a mushroom is **edible (0)** or **poisonous (1)** using various classification models. The dataset contains mushroom features such as cap diameter, stem height, stem width, gill attachment, gill color, stem color, and season.

---

## 📂 Project Structure
├── mushroom_classification.csv   # Dataset
├── assignment.ipynb              # Main Jupyter notebook (your work)
├── assignment_solution.ipynb     # Provided solution reference
└── README.md                     # Documentation


---

## 🔍 Project Workflow

1. **Data Exploration & Preprocessing**
   - Analyzed numeric and categorical features.
   - Conducted t-tests and created boxplots to identify feature importance.
   - Encoded categorical variables for ML models.

2. **Baseline Models**
   - **Logistic Regression**: Accuracy ~64% (weak baseline).
   - **Decision Tree**: Accuracy ~97% (but prone to overfitting).

3. **Model Tuning**
   - Applied pruning (`max_depth`, `min_samples_leaf`) to improve generalization.
   - Used cross-validation and GridSearchCV for hyperparameter tuning.

4. **Ensemble Model**
   - Implemented **Gradient Boosting Classifier**.
   - Tuned parameters: `learning_rate`, `n_estimators`, `max_depth`, etc.
   - Achieved better balance between accuracy and generalization.
   - Extracted **feature importances** for interpretability.

---

## 📊 Results

- **Logistic Regression**: ~64% accuracy (not suitable for this dataset).
- **Decision Tree**: ~97% accuracy (high variance, risk of overfitting).
- **Gradient Boosting**: High accuracy with improved generalization.
- **Key Features**:
  - `stem_width`, `gill_attachment`, and `stem_color` had the highest importance.

---

## 🚀 Tech Stack
- **Python** (pandas, numpy, matplotlib, seaborn, scikit-learn)
- **Jupyter Notebook**

---

## 🏆 Key Learnings
- Logistic Regression struggles with complex, non-linear datasets.
- Decision Trees can easily overfit without pruning.
- Ensemble methods like Gradient Boosting provide better balance and interpretability.
- Feature importance helps explain model decisions.

---

## 📌 Next Steps
- Try **Random Forests** and **XGBoost** for comparison.
- Perform deeper hyperparameter optimization.
- Deploy the model using **Streamlit** or **Flask** for interactive use.

---

👨‍💻 **Author**: Vishal  
🔗 Learning from: Codebasics ML Exercises