
🍷 Wine Clustering Project — Codebasics ML Practice

📘 Overview

This project applies Unsupervised Learning (Clustering) techniques to group wines based on their chemical properties.
It was completed as part of the Codebasics Machine Learning Course.

Using K-Means, Hierarchical Clustering, and DBSCAN, we aim to:
	•	Discover natural clusters among different wine types.
	•	Compare performance using metrics like SSE (Sum of Squared Errors) and Silhouette Score.
	•	Visualize clusters using PCA (Principal Component Analysis) for dimensionality reduction.

⸻

📂 Project Structure

wine-clustering/
│
├── assignment_solution.ipynb      # Main Jupyter Notebook (solution)
├── assignment.ipynb               # Starter/assignment notebook
├── wine_clustering_data.csv       # Dataset used in project
├── wine_clustering_results.csv    # Final dataset with cluster labels (generated)
├── kmeans_centers_original_scale.csv  # KMeans cluster centroids (optional output)
├── README.md                      # Project documentation (this file)
└── images/                        # Folder for plots/screenshots (optional)


⸻

🧠 Key Learning Concepts
	•	Feature Scaling (Standardization)
	•	K-Means Clustering
	•	Elbow Method for optimal K
	•	Silhouette Score for cluster validation
	•	Hierarchical Clustering (Ward linkage)
	•	DBSCAN (Density-Based Clustering)
	•	Dimensionality Reduction using PCA
	•	Cluster Visualization and Interpretation

⸻

⚙️ Tech Stack
	•	Language: Python 3.x
	•	Libraries:
pandas, numpy, matplotlib, seaborn,
scikit-learn, scipy

⸻

🚀 Workflow Summary
	1.	Load Data → wine_clustering_data.csv
Checked shape, missing values, and descriptive statistics.
	2.	Exploratory Data Analysis (EDA)
	•	Histograms to understand feature distributions
	•	Correlation heatmap
	3.	Feature Scaling
Standardized data using StandardScaler to ensure equal weight for all features.
	4.	K-Means Clustering
	•	Applied the Elbow Method to identify optimal K.
	•	Evaluated clusters using Silhouette Score.
	•	Visualized clusters with PCA (2D plot).
	5.	Hierarchical Clustering
	•	Generated Dendrogram using Ward’s linkage.
	•	Compared with K-Means cluster patterns.
	6.	DBSCAN (optional)
	•	Tested density-based clustering for anomaly detection or non-spherical clusters.
	7.	Results & Interpretation
	•	KMeans and Hierarchical both found ~3 clusters.
	•	Visuals showed clear group separation.

⸻

📊 Results

Algorithm	Clusters (K)	Silhouette Score	Notes
K-Means	3	~0.26–0.35	Balanced cluster separation
Agglomerative (Ward)	3	~0.25–0.33	Similar pattern as K-Means
DBSCAN	Varies	Depends on eps	Detects outliers, non-spherical clusters


⸻

📈 Visualizations

Add these screenshots to an /images folder and reference them here:
	1.	Elbow Method Plot
	2.	PCA Cluster Visualization (K-Means)
	3.	Dendrogram (Hierarchical Clustering)
	4.	Agglomerative Cluster Plot

Example markdown embed:

![Elbow Plot](images/elbow_plot.png)
![PCA Clusters](images/pca_clusters.png)
![Dendrogram](images/dendrogram.png)


⸻

💾 Outputs
	•	wine_clustering_results.csv → Dataset with cluster labels (kmeans_label, agg_label, dbscan_label)
	•	kmeans_centers_original_scale.csv → Cluster centroids in original feature scale

⸻

🧩 How to Run Locally
	1.	Clone this repository:

git clone https://github.com/<your-username>/wine-clustering.git
cd wine-clustering


	2.	Install dependencies:

pip install -r requirements.txt


	3.	Launch Jupyter:

jupyter notebook


	4.	Open assignment_solution.ipynb and run all cells.

⸻

🧾 Requirements.txt

pandas
numpy
matplotlib
seaborn
scikit-learn
scipy


⸻

🧠 Insights
	•	Clustering can reveal structure in unlabeled data.
	•	Scaling is crucial — without it, distance-based methods fail.
	•	KMeans and Hierarchical produced consistent grouping → data has well-separated clusters.
	•	DBSCAN helps detect noise or outliers.

⸻

🧑‍💻 Author

Vishal
AI/ML Engineer | 


