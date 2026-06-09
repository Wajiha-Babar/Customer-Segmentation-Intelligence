# Customer Segmentation Intelligence Dashboard

<div align="center">

## Premium Customer Segmentation Using Unsupervised Learning

A complete data science portfolio project that applies **K-Means Clustering**, **PCA**, **t-SNE**, and an interactive **Streamlit dashboard** to segment customers based on income, spending behavior, and demographic patterns.

</div>

---

## Project Overview

This project focuses on **customer segmentation using unsupervised machine learning**. The goal is to divide customers into meaningful groups based on their behavior and then convert those insights into practical marketing strategies.

The project includes:

- A complete **Jupyter Notebook**
- A premium **Streamlit dashboard**
- Exploratory Data Analysis
- K-Means clustering
- PCA and t-SNE visualization
- Cluster profiling
- Marketing strategy recommendations
- Downloadable output files
- GitHub-ready documentation and structure

---

## Objective

The main objective is to cluster customers based on:

- Age
- Annual income
- Spending score

After clustering, each segment is analyzed and assigned a business-friendly name with a suitable marketing strategy.

---

## Dataset Description

The primary dataset used in this project is:

`dataset/Mall_Customers.csv`

### Main Columns

| Column | Description |
|---|---|
| `CustomerID` | Unique customer identifier |
| `Gender` / `Genre` | Customer gender |
| `Age` | Customer age |
| `Annual Income (k$)` | Annual income in thousands of dollars |
| `Spending Score (1-100)` | Score assigned based on customer spending behavior |

The Streamlit dashboard also supports additional datasets for future experimentation:

- `dataset/marketing_campaign.csv`
- `dataset/data.csv`
- `dataset/Wholesale customers data.csv`

---

## Tools and Technologies Used

| Category | Tools |
|---|---|
| Programming Language | Python |
| Data Handling | pandas, numpy |
| Visualization | matplotlib, seaborn, plotly |
| Machine Learning | scikit-learn |
| Clustering | K-Means |
| Dimensionality Reduction | PCA, t-SNE |
| Dashboard | Streamlit |
| Development | Jupyter Notebook, VS Code |
| Version Control | Git, GitHub |

---

## Project Workflow

The project follows a complete data science workflow:

1. Load the customer dataset
2. Inspect dataset shape, columns, and data types
3. Clean column names
4. Handle missing and duplicate values
5. Select useful clustering features
6. Scale features using `StandardScaler`
7. Perform Exploratory Data Analysis
8. Apply the Elbow Method
9. Evaluate clusters using Silhouette Score
10. Train the K-Means model
11. Visualize clusters using PCA and t-SNE
12. Profile each customer segment
13. Assign segment names
14. Suggest marketing strategies
15. Export clustered outputs
16. Build an interactive Streamlit dashboard

---

## Exploratory Data Analysis

The notebook and dashboard include professional visual analysis such as:

- Gender distribution
- Age distribution
- Annual income distribution
- Spending score distribution
- Age vs spending score
- Annual income vs spending score
- Correlation heatmap
- Boxplots for outlier inspection
- Cluster-wise comparison charts

These visualizations help understand customer behavior before applying machine learning.

---

## Machine Learning Method

### K-Means Clustering

K-Means is used to group customers based on similarity in their selected features.

The project evaluates cluster quality using:

- **Elbow Method**
- **Silhouette Score**

The default business-friendly configuration uses **5 clusters**, but the Streamlit dashboard allows users to test different cluster counts from 2 to 10.

---

## Dimensionality Reduction

### PCA

Principal Component Analysis is used to reduce the feature space into two dimensions for cluster visualization.

### t-SNE

t-SNE is used to create a more detailed non-linear visualization of customer clusters.

Both techniques help visually inspect whether customer groups are clearly separated.

---

## Customer Segments

The project assigns business-friendly names to clusters based on average income and spending score.

| Segment | Behavior | Strategy |
|---|---|---|
| Premium Customers | High income and high spending | VIP offers, exclusive access, premium recommendations |
| Potential Customers | High income but low spending | Targeted discounts, demos, premium bundles |
| Deal Seekers | Lower income but high spending | Cashback, loyalty points, seasonal promotions |
| Budget Customers | Lower income and low spending | Low-cost offers, value deals, basic packages |
| Standard Customers | Average income and average spending | Retention campaigns, loyalty cards, cross-sell offers |

---

## Business Insights

This project helps businesses understand:

- Which customers are most valuable
- Which customers need conversion campaigns
- Which customers respond better to discounts
- Which customers should receive premium offers
- Which customer groups need retention strategies

These insights can improve:

- Marketing campaign targeting
- Customer retention
- Sales conversion
- Loyalty program planning
- Budget allocation

---

## Streamlit Dashboard

The project includes a premium dashboard named:

## Customer Segmentation Intelligence Dashboard

### Dashboard Features

- Executive KPI overview
- Dataset preview
- Data quality summary
- Interactive EDA charts
- K-Means clustering controls
- PCA visualization
- t-SNE visualization
- Cluster profile tables
- Marketing strategy cards
- Download buttons for generated CSV files
- Light and dark theme support

---

## Dashboard Preview Sections

The dashboard is divided into the following sections:

1. Executive Overview
2. Dataset Preview
3. EDA Dashboard
4. Clustering Model
5. PCA / t-SNE
6. Cluster Profiles
7. Marketing Strategy
8. Downloads

---

## Project Folder Structure

```text
Customer Segmentation Using Unsupervised Learning/
│
├── dataset/
│   ├── Mall_Customers.csv
│   ├── marketing_campaign.csv
│   ├── data.csv
│   └── Wholesale customers data.csv
│
├── notebooks/
│   └── Customer_Segmentation_KMeans.ipynb
│
├── outputs/
│   ├── clustered_customers.csv
│   ├── cluster_profile.csv
│   └── figures/
│       ├── boxplots_outliers.png
│       ├── correlation_heatmap.png
│       ├── gender_distribution.png
│       ├── kmeans_evaluation.png
│       ├── numeric_distributions.png
│       ├── pca_clusters.png
│       ├── scatter_relationships.png
│       └── tsne_clusters.png
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/Wajiha-Babar/Customer-Segmentation-Intelligence.git
```

### 2. Move into the Project Folder

```bash
cd Customer-Segmentation-Intelligence
```

### 3. Create a Virtual Environment

```powershell
python -m venv .venv
```

### 4. Activate the Virtual Environment

```powershell
.venv\Scripts\activate
```

### 5. Install Required Libraries

```powershell
pip install -r requirements.txt
```

### 6. Open the Jupyter Notebook

```powershell
jupyter notebook
```

### 7. Run the Streamlit Dashboard

```powershell
streamlit run app.py
```

---

## Output Files

The project automatically generates the following files:

| File | Description |
|---|---|
| `outputs/clustered_customers.csv` | Dataset with assigned cluster and segment name |
| `outputs/cluster_profile.csv` | Summary of each cluster |
| `outputs/figures/` | Saved visualizations from analysis |

---

## Results and Findings

The final model groups customers into meaningful business segments. These segments show different income and spending patterns, allowing businesses to design more targeted marketing campaigns.

The segmentation results can be used to:

- Identify premium customers
- Improve customer retention
- Increase sales through personalized offers
- Convert high-income low-spending customers
- Design budget-friendly campaigns for price-sensitive customers

---

## Future Improvements

Possible future improvements include:

- Add DBSCAN clustering
- Add Hierarchical Clustering
- Add Gaussian Mixture Models
- Compare multiple clustering algorithms
- Save trained models using `joblib`
- Add automated PDF report generation
- Connect the dashboard to a database
- Deploy the Streamlit app online
- Add real-time customer data upload support

---

## Repository Purpose

This repository is designed as a practical data science and business analytics portfolio project. It demonstrates the complete workflow from raw customer data to machine learning insights and an interactive business dashboard.

---

## Author

**Wajiha Babar**
