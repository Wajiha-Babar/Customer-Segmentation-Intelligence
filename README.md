# Customer Segmentation Using Unsupervised Learning

## 1. Project Overview
This project segments mall customers with unsupervised machine learning and translates each segment into practical marketing actions. It includes a complete Jupyter Notebook, a premium Streamlit dashboard, saved model outputs, and a clean GitHub-ready structure.

## 2. Objective
Cluster customers based on age, annual income, and spending score, then propose business strategies tailored to each customer segment.

## 3. Dataset Description
The primary dataset is `dataset/Mall_Customers.csv`.

Key columns:
- `CustomerID`: Unique customer identifier
- `Gender` or `Genre`: Customer gender
- `Age`: Customer age
- `Annual Income (k$)`: Annual income in thousands of dollars
- `Spending Score (1-100)`: Mall-assigned spending behavior score

The Streamlit app also supports these future datasets with safe preprocessing:
- `dataset/marketing_campaign.csv`
- `dataset/data.csv`
- `dataset/Wholesale customers data.csv`

## 4. Tools and Libraries Used
- Python
- pandas and numpy
- matplotlib and seaborn
- scikit-learn
- Streamlit
- Plotly
- Jupyter Notebook

## 5. Workflow
1. Load the Mall Customers dataset.
2. Clean column names and handle missing or duplicated values.
3. Select clustering features.
4. Scale features with `StandardScaler`.
5. Explore customer behavior with professional charts.
6. Use the Elbow Method and Silhouette Score to evaluate cluster counts.
7. Fit K-Means clustering.
8. Visualize clusters with PCA and t-SNE.
9. Profile each cluster.
10. Assign segment names and marketing strategies.
11. Save outputs for dashboard and reporting.

## 6. Exploratory Data Analysis
The notebook and dashboard include:
- Gender distribution
- Age distribution
- Annual income distribution
- Spending score distribution
- Age vs spending score
- Annual income vs spending score
- Correlation heatmap
- Boxplots for outlier inspection

## 7. K-Means Clustering
K-Means is used to group customers by similar spending and income behavior. The project evaluates values of K from 2 to 10 using inertia and silhouette score. The default business-friendly solution uses 5 clusters unless silhouette analysis strongly suggests another value.

## 8. PCA and t-SNE Visualization
PCA and t-SNE reduce the scaled feature space into two dimensions so clusters can be inspected visually. Figures are saved in `outputs/figures/`.

## 9. Business Insights
The segmentation helps identify high-value customers, price-sensitive customers, and customers with strong conversion potential. These profiles help teams personalize campaigns, improve retention, and allocate marketing budget more intelligently.

## 10. Marketing Strategies
Segment names are assigned from cluster-level average income and spending score:
- Premium Customers: VIP offers, exclusive membership, early access, and personalized recommendations
- Potential Customers: targeted discounts, product demos, premium bundles, and conversion campaigns
- Deal Seekers: cashback, loyalty points, affordable bundles, and seasonal promotions
- Budget Customers: low-cost offers, basic packages, student discounts, and value deals
- Standard Customers: retention campaigns, loyalty cards, regular promotions, and cross-sell offers

## 11. Streamlit Dashboard
The dashboard is named **Customer Segmentation Intelligence Dashboard** and includes:
- Executive KPI overview
- Dataset preview
- Interactive EDA dashboard
- K-Means clustering controls
- PCA and t-SNE cluster visualization
- Cluster profile tables
- Marketing strategy panel
- Download buttons for generated CSV outputs

## 12. How to Run the Project
Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Open the notebook:

```powershell
jupyter notebook
```

Run the dashboard:

```powershell
streamlit run app.py
```

## 13. Folder Structure
```text
Customer Segmentation Using Unsupervised Learning/
├── dataset/
│   ├── Mall_Customers.csv
│   ├── marketing_campaign.csv
│   ├── data.csv
│   └── Wholesale customers data.csv
├── notebooks/
│   └── Customer_Segmentation_KMeans.ipynb
├── app.py
├── requirements.txt
├── README.md
└── outputs/
    ├── clustered_customers.csv
    ├── cluster_profile.csv
    └── figures/
```

## 14. Results and Findings
The project produces clustered customer records in `outputs/clustered_customers.csv` and a profile summary in `outputs/cluster_profile.csv`. The final segments reveal groups with distinct income and spending patterns, making them useful for campaign targeting and customer relationship management.

## 15. Future Improvements
- Add model comparison with DBSCAN, Gaussian Mixture Models, and Hierarchical Clustering.
- Add automated report generation.
- Store trained scalers and clustering models with joblib.
- Connect the dashboard to a database or cloud storage.
- Add cohort tracking over time if transaction history becomes available.
