# Logistic Regression Multiclass Classification - Dry Bean Dataset

This project is part of a supervised machine learning exercise using the **Dry Bean dataset**.  
The goal is to classify bean types based on various features using **Logistic Regression**.

---

## 📂 Project Structure
- `assignment.ipynb` → Problem statement and guided steps
- `assignment_solution.ipynb` → Full solution with code, visualizations, and explanations
- `dry_bean_dataset.xlsx` → Dataset file
- `.DS_Store` → System file (can be ignored)

---

## 📊 Dataset
The dataset contains multiple features of dry beans such as:
- **Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRatio, Eccentricity, ConvexArea, EquivalentDiameter, Solidity, Roundness, Compactness, Shape factors**  
- **Class**: Target variable with 7 categories of bean types (Seker, Barbunya, Bombay, Cali, Dermason, Horoz, Sira).

---

## 🚀 Workflow
1. **Data Preparation & Exploration**
   - Import dataset, check shape, missing values, class distribution
   - Visualize feature distributions per class

2. **Data Preprocessing**
   - Feature-target split
   - Train-test split (70-30)
   - Standard scaling of features

3. **Model Training**
   - Logistic Regression (baseline)
   - Logistic Regression with tuned parameters:
     - `max_iter=300`
     - `C=0.5`
     - `tol=0.001`
     - `class_weight='balanced'`

4. **Evaluation**
   - Accuracy score
   - Classification report (Precision, Recall, F1-score, Support)
   - Confusion matrix (heatmap)

---

## 📈 Results
- **Baseline Logistic Regression Accuracy**: ~92%
- Strong performance across most bean types  
- Some misclassification observed in **Dermason** vs **Sira** classes

---

## 🔧 Requirements
- Python 3.8+
- Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

Install dependencies:
```bash
pip install -r requirements.txt