# Weather Classification using Support Vector Machines (SVM)

This project demonstrates the workflow of a supervised machine learning classification task.  
We classify weather types (Sunny, Rainy, Snowy, Cloudy) using **Support Vector Machines (SVM)** with different kernels, hyperparameter tuning, and sklearn pipelines.

---

## 📂 Project Structure
Codebasics_ML_Exercises/
│── supervised_machine_learning_classification/
│   ├── assignment.ipynb              # Main notebook (work-in-progress)
│   ├── assignment_solution.ipynb     # Reference solution notebook
│   ├── weather_classification_data.csv  # Dataset
│   └── README.md                     # Project documentation

---

## 🛠️ Workflow

### **Task 1: Data Preparation & Exploration**
- Loaded dataset (`weather_classification_data.csv`).
- Checked structure, missing values, duplicates, and categorical distributions.

### **Task 2: Data Transformation**
- One-hot encoded categorical variables: `cloud_cover`, `location`, `season`.
- Scaled numerical features using `StandardScaler`.

### **Task 3: Model Training (Linear SVM)**
- Encoded target labels (`weather_type`) with `LabelEncoder`.
- Train-test split (70/30, stratified).
- Trained **SVM with linear kernel**.
- Accuracy: ~88.6%.

### **Task 4: Model Training (RBF SVM)**
- Trained **SVM with RBF kernel**.
- Accuracy: ~90.9%.

### **Task 5: Hyperparameter Tuning**
- Tuned hyperparameters (`C`, `gamma`) using GridSearchCV.
- Best parameters: `C=5`, `gamma=0.1`, `kernel='rbf'`.
- Accuracy improved to ~91.4%.

### **Task 6: Sklearn Pipeline**
- Built an ML pipeline: `StandardScaler` → `SVC (RBF)`.
- End-to-end automation of preprocessing + training.
- Evaluated with classification report & confusion matrix.

---

## 📊 Results

| Model            | Accuracy |
|------------------|----------|
| Linear SVM       | 88.6%    |
| RBF SVM (default)| 90.9%    |
| RBF SVM (tuned)  | **91.4%** |

- Best performance from **RBF kernel SVM with tuned hyperparameters**.
- Balanced precision/recall across all weather classes.

---

## 🔑 Key Learnings
- Why **scaling** is essential for SVMs.
- Role of **C** (regularization strength) and **gamma** (decision boundary flexibility).
- One-hot encoding for categorical features.
- Using pipelines for reproducibility and deployment.

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone <your-repo-link>
   cd Codebasics_ML_Exercises/supervised_machine_learning_classification


pip install -r requirements.txt
jupyter notebook assignment.ipynb