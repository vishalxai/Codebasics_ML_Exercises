Predicting Daily Temperature from Sunlight & Humidity

A compact end-to-end regression project using linear regression to predict daily_temperature (°C) from hours_sunlight and humidity_level.

🧠 Problem

You’re a data scientist at a meteorological firm. Given a daily log of:
	•	hours_sunlight — total sunlight hours in a day
	•	humidity_level — humidity (%)
	•	daily_temperature — temperature (°C)

Build, evaluate, and explain a linear model to predict daily temperature.

📂 Dataset
	•	File: weather_data.csv
	•	Columns: hours_sunlight, humidity_level, daily_temperature
	•	No missing values in the provided file (verify on load).
	•	Small tabular dataset (≈ 50 rows).

If you change the dataset, re-run the sanity checks below.

🔧 Environment
	•	Python ≥ 3.9
	•	Dependencies: pandas, numpy, scikit-learn, matplotlib
    pip install -r requirements.txt

    requirements.txt
    pandas
numpy
scikit-learn
matplotlib

🚀 Quick Start
	1.	Clone this repo and open the notebook:

    git clone <your-repo-url>
cd <repo>
jupyter notebook assignment.ipynb

