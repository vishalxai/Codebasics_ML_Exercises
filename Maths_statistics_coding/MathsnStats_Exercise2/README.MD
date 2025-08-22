# Fitness Tracker Data Analysis – Outlier Detection (Codebasics Exercise)

## 📌 Problem Statement
We are working with fitness tracker data from 50 individuals.  
Dataset: `fitness_data.xlsx`  
Columns:
- `name` – individual’s name  
- `steps_taken` – daily steps  
- `calories_burned` – daily calories burned  
- `sleep_duration(hours)` – sleep in hours  
- `water_intake(ounces)` – water consumption  

Goal: Perform exploratory data analysis (EDA), compute summary statistics, and detect outliers using **IQR and boxplots**.

---

## 📊 Steps Performed

### 1. Data Import & Cleaning
- Loaded Excel file using pandas.  
- Checked schema, data types, missing values, duplicates.  
- Standardized column names.

### 2. Descriptive Statistics
- Calculated `mean`, `median`, `min`, `max` for all numeric columns.  
- Compared **mean vs median** to check for skewness.  
- Example: `steps_taken` had median ~10,000, mean slightly shifted due to high outliers.

### 3. Range & IQR
- Range = `max – min` (sensitive to outliers).  
- IQR = `Q3 – Q1` (robust to outliers).  
- Example: Sleep hours had **IQR = 0.5** → very consistent for most individuals.

### 4. Visualization
- Created **boxplots** for steps and other metrics.  
- Outliers clearly visible:  
  - e.g., `steps_taken` > 15,000  
  - `sleep_duration` = 18 hrs (impossible) vs 4 hrs (rare but possible).

### 5. Outlier Detection (IQR Method)
- Used IQR rule:  
  - Lower bound = Q1 – 1.5 × IQR  
  - Upper bound = Q3 + 1.5 × IQR  
- Flagged outliers in `sleep_duration(hours)`.  
- Example:  
  - Elena → 18 hrs sleep (error, remove)  
  - Siddharth → 4 hrs sleep (valid but rare, keep/flag).  
- Wrote a **general function** `detect_outliers_iqr()` to reuse across columns.

---

## 🧠 Key Learnings
- **Mean vs Median**: Median is more robust in presence of outliers.  
- **IQR vs Range**: IQR reflects the stable bulk of the data, range highlights extremes.  
- **Boxplots**: Fast way to spot outliers visually.  
- **Business Context Matters**: Not all outliers should be removed. Some contain valuable information.  

---

## 🛠️ Tech Stack
- Python 3.x  
- Pandas, Numpy, Matplotlib  

---

## 📂 Repository Structure