
# Day 4 - Advanced Customer Behaviour Clustering

An interactive machine learning application that discovers customer behaviour patterns using K-Means and DBSCAN clustering.

## Features

- Built-in synthetic customer behaviour dataset
- Upload custom CSV datasets
- Automated dataset profiling
- Missing-value handling
- Feature selection and standardization
- K-Means clustering
- DBSCAN clustering
- PCA-based 2D cluster visualization
- Silhouette score evaluation
- Cluster profiling and heatmaps
- K-Means vs DBSCAN comparison
- Download clustered datasets and profiles

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd day_04_customer_behaviour_clustering
```

### 2. Create a virtual environment

```powershell
py -3.10 -m venv .venv
```

### 3. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
python -m pip install --upgrade pip --timeout 600 --retries 10
python -m pip install -r requirements.txt --no-cache-dir --timeout 600 --retries 10
```

### 5. Run the application

```powershell
streamlit run app.py
```

## Dataset

The project includes a built-in synthetic dataset. No external download is required.

For custom datasets, upload a CSV containing at least two useful numeric behaviour features.

## Algorithms

### K-Means
Partitions observations into a predefined number of clusters.

### DBSCAN
Identifies density-connected groups and can label observations as noise.

### PCA
Projects scaled features into two dimensions for visualization.

## Evaluation

The application calculates the silhouette score when the resulting labels support it. The score should be interpreted alongside cluster distribution and domain context.

## Project Goal

Demonstrate practical skills in unsupervised machine learning, data preprocessing, dimensionality reduction, model evaluation and interactive dashboard development.
