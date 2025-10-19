# Country Clustering Analysis

## Overview
This repository contains a Jupyter notebook (`assignment.ipynb`) that performs an unsupervised learning analysis to cluster countries based on socio-economic and health factors. The project aims to assist HELP International in strategically allocating $10 million by identifying country development levels using data from `country_data.csv`.

## Dataset
- **Source**: [Kaggle - Unsupervised Learning on Country Data](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data/data)
- **License**: MIT License
- **Features**:
  - `country`: Name of the country
  - `child_mort`: Child mortality rate (deaths under 5 per 1000 live births)
  - `exports`: Exports as % of GDP
  - `health`: Health expenditure as % of GDP
  - `imports`: Imports as % of GDP
  - `income`: Net income per person (USD)
  - `inflation`: Inflation rate (%)
  - `life_expec`: Life expectancy at birth (years)
  - `total_fer`: Total fertility rate (children per woman)
  - `gdpp`: GDP per capita (USD)

## Project Structure
- `assignment.ipynb`: Main Jupyter notebook with code and analysis.
- `country_data.csv`: Dataset containing country statistics (included in the document).

## Methodology
1. **Data Preparation and Exploration**:
   - Loaded and explored the dataset using pandas.
   - Visualized distributions (histograms) and relationships (pairplots) for key features.
2. **Initial K-Means Clustering**:
   - Standardized features using `StandardScaler`.
   - Applied K-Means with 4 clusters, analyzed inertia and centroids, and visualized clusters.
3. **Elbow Method**:
   - Determined the optimal number of clusters (1–20) using the Elbow Method.
   - Trained K-Means with the optimal number of clusters.
4. **Hierarchical Clustering**:
   - Applied Agglomerative Clustering with 6 clusters (cosine metric, complete linkage).
   - Plotted a dendrogram and pairplot to visualize results.

## Requirements
- Python 3.x
- Libraries:
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `numpy`
  - `scikit-learn`
  - `scipy`
- Install via: `pip install pandas matplotlib seaborn numpy scikit-learn scipy`

## Usage
1. Clone the repository: `git clone <repository-url>`
2. Navigate to the directory: `cd <repository-directory>`
3. Install dependencies: `pip install -r requirements.txt` (create `requirements.txt` with listed libraries if needed)
4. Open and run `assignment.ipynb` in Jupyter Notebook or a compatible environment.

## Results
- **K-Means**: Initial 4 clusters, refined to an optimal number (e.g., 3 or 4) via the Elbow Method.
- **Hierarchical**: 6 clusters with visualizations showing country groupings.
- Clusters reflect development levels, with insights into aid allocation (e.g., high `child_mort`, low `gdpp` countries).

## Contributing
Feel free to fork this repository, suggest improvements, or report issues via GitHub.

## License
This project is for educational purposes and uses the dataset under the MIT License.

## Acknowledgments
- Dataset credits: Mark Otto, Andrew Fong (via Kaggle).
- Guided by xAI's Grok for step-by-step learning.

---
