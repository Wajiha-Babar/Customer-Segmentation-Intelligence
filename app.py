from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Segmentation Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)


# ============================================================
# DATASET OPTIONS
# ============================================================

DEFAULT_DATASETS = {
    "Mall Customers": DATASET_DIR / "Mall_Customers.csv",
    "Marketing Campaign": DATASET_DIR / "marketing_campaign.csv",
    "Generic Data": DATASET_DIR / "data.csv",
    "Wholesale Customers": DATASET_DIR / "Wholesale customers data.csv",
}


def discover_datasets() -> dict[str, Path]:
    """Find all CSV datasets inside the dataset folder."""
    datasets: dict[str, Path] = {}

    for name, path in DEFAULT_DATASETS.items():
        if path.exists():
            datasets[name] = path

    if DATASET_DIR.exists():
        for file in DATASET_DIR.glob("*.csv"):
            custom_name = file.stem.replace("_", " ").replace("-", " ").title()
            if file not in datasets.values():
                datasets[custom_name] = file

    return datasets


DATASET_OPTIONS = discover_datasets()


# ============================================================
# BUSINESS STRATEGIES
# ============================================================

STRATEGIES = {
    "Premium Customers": {
        "behavior": "High income with high spending activity.",
        "interpretation": "This segment is highly valuable and already strongly engaged.",
        "strategy": "Offer VIP membership, exclusive launches, premium bundles, private sales, early access, and personalized recommendations.",
    },
    "Potential Customers": {
        "behavior": "High income but comparatively low spending.",
        "interpretation": "These customers have strong purchasing power but need better motivation to spend.",
        "strategy": "Use personalized offers, product demos, targeted discounts, premium bundles, and conversion-focused campaigns.",
    },
    "Deal Seekers": {
        "behavior": "Lower income with high spending activity.",
        "interpretation": "This segment responds well to excitement, value, promotions, and loyalty benefits.",
        "strategy": "Promote cashback, loyalty points, seasonal sales, affordable bundles, limited-time offers, and reward programs.",
    },
    "Budget Customers": {
        "behavior": "Lower income with lower spending activity.",
        "interpretation": "This segment is price-sensitive and needs simple value-based offers.",
        "strategy": "Recommend low-cost packages, basic offers, student discounts, value deals, and budget-friendly promotions.",
    },
    "Standard Customers": {
        "behavior": "Average income and average spending behavior.",
        "interpretation": "This is a stable mainstream group with potential for retention and moderate upselling.",
        "strategy": "Use regular promotions, loyalty cards, retention campaigns, cross-sell suggestions, and personalized reminders.",
    },
}


# ============================================================
# THEME CSS
# ============================================================

def apply_theme(theme: str) -> dict[str, str]:
    """Apply a readable navy/white luxury theme for light and dark modes."""
    if theme == "Dark":
        colors = {
            "app_bg": "#07111F",
            "app_bg_2": "#0B1728",
            "sidebar_bg_1": "#020B18",
            "sidebar_bg_2": "#061A3A",
            "card_bg": "#0F2137",
            "card_border": "#263E63",
            "text": "#F8FAFC",
            "muted": "#CBD5E1",
            "hero_1": "#061A3A",
            "hero_2": "#173C78",
            "accent": "#E4C76F",
            "button": "#E4C76F",
            "button_text": "#061A3A",
            "tab": "#CBD5E1",
            "tab_active": "#E4C76F",
            "input_bg": "#FFFFFF",
            "input_text": "#061A3A",
            "success_bg": "#082F2A",
            "warning_bg": "#3B2F0B",
        }
    else:
        colors = {
            "app_bg": "#F7F9FD",
            "app_bg_2": "#EEF3FA",
            "sidebar_bg_1": "#061A3A",
            "sidebar_bg_2": "#0B254F",
            "card_bg": "#FFFFFF",
            "card_border": "#E2E8F0",
            "text": "#061A3A",
            "muted": "#475569",
            "hero_1": "#061A3A",
            "hero_2": "#173C78",
            "accent": "#C9A227",
            "button": "#061A3A",
            "button_text": "#FFFFFF",
            "tab": "#334155",
            "tab_active": "#E11D48",
            "input_bg": "#FFFFFF",
            "input_text": "#061A3A",
            "success_bg": "#ECFDF5",
            "warning_bg": "#FFFBEB",
        }

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        .stApp {{
            background: linear-gradient(180deg, {colors['app_bg']} 0%, {colors['app_bg_2']} 100%);
            color: {colors['text']};
        }}

        .block-container {{
            padding-top: 1.25rem;
            padding-bottom: 2.5rem;
            max-width: 1380px;
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {colors['sidebar_bg_1']} 0%, {colors['sidebar_bg_2']} 100%);
        }}

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown {{
            color: #FFFFFF !important;
        }}

        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div,
        [data-baseweb="textarea"] > div {{
            background-color: {colors['input_bg']} !important;
            color: {colors['input_text']} !important;
            border-radius: 10px !important;
            border: 1px solid #CBD5E1 !important;
        }}

        [data-baseweb="select"] span,
        [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea {{
            color: {colors['input_text']} !important;
        }}

        [data-baseweb="popover"],
        [data-baseweb="menu"] {{
            background-color: {colors['input_bg']} !important;
            color: {colors['input_text']} !important;
        }}

        [data-baseweb="menu"] li {{
            color: {colors['input_text']} !important;
        }}

        [data-baseweb="tag"] {{
            background-color: #EAF1FF !important;
            color: #061A3A !important;
            border-radius: 8px !important;
        }}

        [data-baseweb="tag"] span {{
            color: #061A3A !important;
            font-weight: 700 !important;
        }}

        .hero {{
            background: linear-gradient(135deg, {colors['hero_1']} 0%, {colors['hero_2']} 100%);
            border-radius: 20px;
            padding: 36px 38px;
            color: white;
            margin-bottom: 22px;
            border: 1px solid rgba(228, 199, 111, 0.42);
            box-shadow: 0 18px 42px rgba(6, 26, 58, 0.22);
        }}

        .hero h1 {{
            margin: 0;
            font-size: clamp(2rem, 4vw, 3.35rem);
            font-weight: 800;
            letter-spacing: -0.04em;
            color: #FFFFFF;
        }}

        .hero p {{
            margin-top: 12px;
            color: #DCE7F7;
            font-size: 1rem;
            max-width: 960px;
            line-height: 1.7;
        }}

        .metric-card {{
            background: {colors['card_bg']};
            border: 1px solid {colors['card_border']};
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 12px 30px rgba(8, 31, 66, 0.12);
            min-height: 124px;
        }}

        .metric-label {{
            color: {colors['muted']};
            font-size: 0.78rem;
            text-transform: uppercase;
            font-weight: 800;
            letter-spacing: 0.05em;
        }}

        .metric-value {{
            color: {colors['text']};
            font-size: 1.85rem;
            font-weight: 800;
            line-height: 1.15;
            margin-top: 10px;
        }}

        .card {{
            background: {colors['card_bg']};
            border: 1px solid {colors['card_border']};
            border-radius: 16px;
            padding: 22px;
            box-shadow: 0 12px 30px rgba(8, 31, 66, 0.10);
            margin-bottom: 16px;
            color: {colors['text']};
        }}

        .section-title {{
            color: {colors['text']};
            font-size: 1.2rem;
            font-weight: 800;
            margin-bottom: 12px;
        }}

        .small-note {{
            color: {colors['muted']};
            font-size: 0.92rem;
            line-height: 1.65;
        }}

        .strategy-name {{
            color: {colors['text']};
            font-weight: 800;
            font-size: 1.1rem;
            margin-bottom: 8px;
        }}

        .strategy-pill {{
            display: inline-block;
            background: rgba(228, 199, 111, 0.18);
            color: {colors['accent']};
            padding: 5px 12px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 800;
            margin-bottom: 12px;
            border: 1px solid rgba(228, 199, 111, 0.42);
        }}

        .explain-box {{
            background: {colors['card_bg']};
            border-left: 5px solid {colors['accent']};
            border-radius: 12px;
            padding: 18px 20px;
            margin: 14px 0 18px 0;
            border-top: 1px solid {colors['card_border']};
            border-right: 1px solid {colors['card_border']};
            border-bottom: 1px solid {colors['card_border']};
        }}

        div[data-testid="stDataFrame"] {{
            border: 1px solid {colors['card_border']};
            border-radius: 12px;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            border-bottom: 1px solid {colors['card_border']};
        }}

        .stTabs [data-baseweb="tab"] {{
            color: {colors['tab']};
            font-weight: 650;
        }}

        .stTabs [aria-selected="true"] {{
            color: {colors['tab_active']} !important;
            border-bottom: 3px solid {colors['tab_active']} !important;
        }}

        .stDownloadButton > button,
        .stButton > button {{
            background: {colors['button']} !important;
            color: {colors['button_text']} !important;
            border-radius: 10px !important;
            border: 0 !important;
            font-weight: 800 !important;
            padding: 0.65rem 1rem !important;
        }}

        h1, h2, h3, h4, h5, h6, p, span, label {{
            color: inherit;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    return colors


# ============================================================
# DATA FUNCTIONS
# ============================================================

@st.cache_data(show_spinner=False)
def load_data_from_path(path_string: str) -> pd.DataFrame:
    """Load CSV data and automatically detect common delimiters."""
    path = Path(path_string)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        sample = file.read(4096)

    separators = {
        ",": sample.count(","),
        "\t": sample.count("\t"),
        ";": sample.count(";"),
    }

    separator = max(separators, key=separators.get)
    return pd.read_csv(path, sep=separator)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names, create useful features, fill missing values, and remove duplicates."""
    data = df.copy()
    data.columns = [str(col).strip() for col in data.columns]

    rename_map = {
        "Genre": "Gender",
        "gender": "Gender",
        "Annual Income (k$)": "Annual_Income",
        "Annual Income": "Annual_Income",
        "Income": "Annual_Income",
        "income": "Annual_Income",
        "Spending Score (1-100)": "Spending_Score",
        "Spending Score": "Spending_Score",
        "Customer ID": "CustomerID",
        "CustomerID": "CustomerID",
        "Customer Id": "CustomerID",
        "ID": "CustomerID",
        "Id": "CustomerID",
    }

    data = data.rename(columns={old: new for old, new in rename_map.items() if old in data.columns})

    # Marketing Campaign dataset: convert birth year into age.
    if "Year_Birth" in data.columns and "Age" not in data.columns:
        data["Age"] = 2026 - pd.to_numeric(data["Year_Birth"], errors="coerce")

    # Marketing Campaign dataset: create a total spending score from product spending columns.
    if "Spending_Score" not in data.columns:
        spending_cols = [
            col for col in data.columns
            if str(col).startswith("Mnt") or "spending" in str(col).lower() or "score" in str(col).lower()
        ]

        if spending_cols:
            numeric_spending = data[spending_cols].apply(pd.to_numeric, errors="coerce")
            data["Spending_Score"] = numeric_spending.sum(axis=1)

    # Clean text columns.
    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = data[col].astype(str).str.strip()

    # Clean numeric columns.
    for col in data.select_dtypes(include=np.number).columns:
        data[col] = data[col].replace([np.inf, -np.inf], np.nan)
        median_value = data[col].median()
        if pd.isna(median_value):
            median_value = 0
        data[col] = data[col].fillna(median_value)

    # Clean categorical columns.
    for col in data.select_dtypes(exclude=np.number).columns:
        mode_value = data[col].mode()
        fill_value = mode_value.iloc[0] if not mode_value.empty else "Unknown"
        data[col] = data[col].fillna(fill_value)

    data = data.drop_duplicates().reset_index(drop=True)
    return data


def get_default_features(data: pd.DataFrame) -> list[str]:
    """Select default clustering features."""
    preferred = [
        col for col in ["Age", "Annual_Income", "Spending_Score"]
        if col in data.columns and pd.api.types.is_numeric_dtype(data[col])
    ]

    if len(preferred) >= 2:
        return preferred

    excluded = {"customerid", "customer_id", "id", "z_costcontact", "z_revenue"}
    numeric_cols = [
        col for col in data.select_dtypes(include=np.number).columns
        if str(col).lower() not in excluded
    ]

    return numeric_cols[:6]


def scale_features(data: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data[features])
    return scaled, scaler


def run_kmeans(scaled_features: np.ndarray, k: int) -> tuple[KMeans, np.ndarray, float]:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(scaled_features)

    if len(set(labels)) > 1 and len(scaled_features) > k:
        score = silhouette_score(scaled_features, labels)
    else:
        score = np.nan

    return model, labels, score


def assign_segment_names(profile: pd.DataFrame) -> dict[int, str]:
    """Assign human-readable segment names using income and spending behavior."""
    numeric_cols = profile.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ["Cluster", "Customer_Count"]]

    if not numeric_cols:
        return {int(cluster): "Standard Customers" for cluster in profile["Cluster"]}

    income_col = "Annual_Income" if "Annual_Income" in profile.columns else numeric_cols[0]
    spending_col = "Spending_Score" if "Spending_Score" in profile.columns else numeric_cols[-1]

    income_median = profile[income_col].median()
    spending_median = profile[spending_col].median()

    names: dict[int, str] = {}

    for _, row in profile.iterrows():
        cluster = int(row["Cluster"])
        high_income = row[income_col] >= income_median
        high_spending = row[spending_col] >= spending_median

        if high_income and high_spending:
            segment = "Premium Customers"
        elif high_income and not high_spending:
            segment = "Potential Customers"
        elif not high_income and high_spending:
            segment = "Deal Seekers"
        elif not high_income and not high_spending:
            segment = "Budget Customers"
        else:
            segment = "Standard Customers"

        names[cluster] = segment

    return names


def create_cluster_profile(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    profile = data.groupby("Cluster")[features].mean().round(2)
    profile["Customer_Count"] = data.groupby("Cluster").size()
    profile = profile.reset_index()

    segment_map = assign_segment_names(profile)
    profile["Segment_Name"] = profile["Cluster"].map(segment_map)

    return profile


def generate_marketing_strategy(segment_name: str) -> dict[str, str]:
    return STRATEGIES.get(segment_name, STRATEGIES["Standard Customers"])


# ============================================================
# VISUAL FUNCTIONS
# ============================================================

def get_plot_template(theme: str) -> str:
    return "plotly_dark" if theme == "Dark" else "plotly_white"


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_corr_heatmap(data: pd.DataFrame, features: list[str], theme: str) -> go.Figure:
    corr = data[features].corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="Blues",
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation"),
        )
    )

    fig.update_layout(
        title="Correlation Heatmap",
        height=460,
        template=get_plot_template(theme),
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def safe_histogram(data: pd.DataFrame, x: str, title: str, theme: str) -> go.Figure:
    fig = px.histogram(
        data,
        x=x,
        nbins=25,
        title=title,
        color_discrete_sequence=["#1E5AA8"],
        template=get_plot_template(theme),
    )
    fig.update_layout(height=420)
    return fig


def safe_scatter(
    data: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    title: str,
    theme: str,
    hover_data: list[str] | None = None,
) -> go.Figure:
    fig = px.scatter(
        data,
        x=x,
        y=y,
        color=color,
        hover_data=hover_data,
        title=title,
        color_discrete_sequence=px.colors.qualitative.Bold,
        template=get_plot_template(theme),
    )
    fig.update_traces(marker=dict(size=10, opacity=0.85))
    fig.update_layout(height=520)
    return fig


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Control Center")

    selected_theme = st.radio(
        "Dashboard Theme",
        ["Light", "Dark"],
        horizontal=True,
        index=0,
    )

colors = apply_theme(selected_theme)

with st.sidebar:
    if not DATASET_OPTIONS:
        st.error("No CSV dataset found inside the dataset folder.")
        st.stop()

    selected_dataset_name = st.selectbox(
        "Dataset",
        list(DATASET_OPTIONS.keys()),
        index=0,
    )

    selected_dataset_path = DATASET_OPTIONS[selected_dataset_name]
    st.caption(f"Selected file: {selected_dataset_path.name}")

    st.markdown("---")
    st.caption(
        "The dashboard automatically cleans the data, selects numeric features, applies K-Means, and generates marketing strategies."
    )


# ============================================================
# LOAD + PROCESS DATA
# ============================================================

try:
    raw_df = load_data_from_path(str(selected_dataset_path))
except Exception as error:
    st.error(f"Unable to load dataset: {error}")
    st.stop()

clean_df = clean_data(raw_df)
default_features = get_default_features(clean_df)

if len(default_features) < 2:
    st.error("At least two numeric features are required for clustering.")
    st.stop()

with st.sidebar:
    numeric_feature_options = [
        col for col in clean_df.select_dtypes(include=np.number).columns
        if str(col).lower() not in ["customerid", "customer_id", "id"]
    ]

    selected_features = st.multiselect(
        "Clustering Features",
        options=numeric_feature_options,
        default=default_features,
        help="Choose numeric columns that should be used by K-Means clustering.",
    )

    if len(selected_features) < 2:
        st.warning("Please select at least two numeric features.")
        st.stop()

    if len(clean_df) < 3:
        st.error("Dataset must contain at least 3 rows for clustering.")
        st.stop()

    max_possible_k = min(10, len(clean_df) - 1)

    selected_k = st.slider(
        "Number of Clusters",
        min_value=2,
        max_value=max_possible_k,
        value=min(5, max_possible_k),
    )

scaled_features, scaler = scale_features(clean_df, selected_features)
model, labels, silhouette = run_kmeans(scaled_features, selected_k)

clustered_df = clean_df.copy()
clustered_df["Cluster"] = labels

profile_df = create_cluster_profile(clustered_df, selected_features)

segment_lookup = dict(zip(profile_df["Cluster"], profile_df["Segment_Name"]))
clustered_df["Segment_Name"] = clustered_df["Cluster"].map(segment_lookup)

clustered_output_path = OUTPUT_DIR / "clustered_customers.csv"
profile_output_path = OUTPUT_DIR / "cluster_profile.csv"

clustered_df.to_csv(clustered_output_path, index=False)
profile_df.to_csv(profile_output_path, index=False)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>Customer Segmentation Intelligence Dashboard</h1>
        <p>
        Premium unsupervised learning dashboard for discovering customer groups,
        profiling behavior, visualizing clusters, and converting data insights into
        practical marketing strategy.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TABS
# ============================================================

tabs = st.tabs(
    [
        "Executive Overview",
        "Dataset Preview",
        "EDA Dashboard",
        "Clustering Model",
        "PCA / t-SNE",
        "Cluster Profiles",
        "Marketing Strategy",
        "Downloads",
    ]
)


# ============================================================
# TAB 1: EXECUTIVE OVERVIEW
# ============================================================

with tabs[0]:
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        metric_card("Total Customers", f"{len(clustered_df):,}")

    with c2:
        metric_card("Average Age", f"{clustered_df['Age'].mean():.1f}" if "Age" in clustered_df.columns else "N/A")

    with c3:
        metric_card(
            "Average Income",
            f"{clustered_df['Annual_Income'].mean():.1f}" if "Annual_Income" in clustered_df.columns else "N/A",
        )

    with c4:
        metric_card(
            "Avg Spending Score",
            f"{clustered_df['Spending_Score'].mean():.1f}" if "Spending_Score" in clustered_df.columns else "N/A",
        )

    with c5:
        metric_card("Clusters", str(selected_k))

    st.markdown("### Segment Distribution")

    fig_segment = px.bar(
        profile_df,
        x="Segment_Name",
        y="Customer_Count",
        color="Segment_Name",
        text="Customer_Count",
        title="Customer Count by Segment",
        color_discrete_sequence=px.colors.qualitative.Bold,
        template=get_plot_template(selected_theme),
    )
    fig_segment.update_layout(height=500, xaxis_title="Segment", yaxis_title="Customers")
    st.plotly_chart(fig_segment, use_container_width=True)

    st.markdown(
        """
        <div class="card">
            <div class="section-title">Executive Summary</div>
            <p>
            This dashboard applies K-Means clustering to group customers based on selected behavioral and demographic features.
            The resulting segments support targeted marketing, loyalty campaigns, premium offers, and retention planning.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# TAB 2: DATASET PREVIEW
# ============================================================

with tabs[1]:
    st.markdown("### Dataset Preview & Data Quality Summary")

    st.markdown(
        f"""
        <div class="card">
            <div class="section-title">Current Dataset</div>
            <p>
            You are currently using <strong>{selected_dataset_name}</strong>. This dataset is cleaned and converted into
            numeric features for customer segmentation using K-Means clustering.
            </p>
            <p class="small-note">File name: <strong>{selected_dataset_path.name}</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_rows = raw_df.shape[0]
    total_columns = raw_df.shape[1]
    total_missing = int(raw_df.isna().sum().sum())
    duplicate_rows = int(raw_df.duplicated().sum())
    numeric_columns = clean_df.select_dtypes(include=np.number).shape[1]

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        metric_card("Rows", f"{total_rows:,}")

    with c2:
        metric_card("Columns", f"{total_columns:,}")

    with c3:
        metric_card("Missing Values", f"{total_missing:,}")

    with c4:
        metric_card("Duplicate Rows", f"{duplicate_rows:,}")

    with c5:
        metric_card("Numeric Columns", f"{numeric_columns:,}")

    st.markdown(
        """
        <div class="explain-box">
            <div class="section-title">What These Values Mean</div>
            <p><strong>Rows:</strong> Total customer records in the dataset.</p>
            <p><strong>Columns:</strong> Total customer attributes such as age, income, education, purchases, spending, or campaign response.</p>
            <p><strong>Missing Values:</strong> Empty cells that need treatment before analysis. The app fills numeric missing values using the median.</p>
            <p><strong>Duplicate Rows:</strong> Repeated customer records. The app removes duplicates during preprocessing.</p>
            <p><strong>Numeric Columns:</strong> Columns that can be used for machine learning and clustering.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="card"><div class="section-title">Clean Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(clean_df.head(20), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="section-title">Column Summary</div>', unsafe_allow_html=True)
    column_summary = pd.DataFrame(
        {
            "Column Name": raw_df.columns,
            "Data Type": raw_df.dtypes.astype(str).values,
            "Missing Values": raw_df.isna().sum().values,
            "Unique Values": raw_df.nunique(dropna=True).values,
        }
    )
    st.dataframe(column_summary, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="section-title">Missing Values Report</div>', unsafe_allow_html=True)
    missing_only = column_summary[column_summary["Missing Values"] > 0]

    if missing_only.empty:
        st.success("No missing values found in this dataset.")
    else:
        st.warning("Some columns contain missing values. These are handled during preprocessing.")
        st.dataframe(missing_only, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="section-title">Selected Clustering Features</div>', unsafe_allow_html=True)
    selected_feature_summary = pd.DataFrame(
        {
            "Selected Feature": selected_features,
            "Mean": [round(clean_df[col].mean(), 2) for col in selected_features],
            "Minimum": [round(clean_df[col].min(), 2) for col in selected_features],
            "Maximum": [round(clean_df[col].max(), 2) for col in selected_features],
            "Standard Deviation": [round(clean_df[col].std(), 2) for col in selected_features],
        }
    )
    st.dataframe(selected_feature_summary, use_container_width=True)
    st.markdown(
        """
        <p class="small-note">
        These selected features are scaled using StandardScaler before applying K-Means clustering.
        Scaling is important because K-Means calculates distance between customers.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="section-title">Basic Statistical Summary</div>', unsafe_allow_html=True)
    st.dataframe(clean_df[selected_features].describe().round(2), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# TAB 3: EDA DASHBOARD
# ============================================================

with tabs[2]:
    c1, c2 = st.columns(2)

    with c1:
        if "Gender" in clustered_df.columns:
            fig_gender = px.histogram(
                clustered_df,
                x="Gender",
                color="Gender",
                title="Gender Distribution",
                color_discrete_sequence=["#061A3A", "#C9A227", "#1E5AA8"],
                template=get_plot_template(selected_theme),
            )
            fig_gender.update_layout(height=420)
            st.plotly_chart(fig_gender, use_container_width=True)

        if "Age" in clustered_df.columns:
            st.plotly_chart(
                safe_histogram(clustered_df, "Age", "Age Distribution", selected_theme),
                use_container_width=True,
            )

    with c2:
        if "Annual_Income" in clustered_df.columns:
            st.plotly_chart(
                safe_histogram(clustered_df, "Annual_Income", "Annual Income Distribution", selected_theme),
                use_container_width=True,
            )

        if "Spending_Score" in clustered_df.columns:
            st.plotly_chart(
                safe_histogram(clustered_df, "Spending_Score", "Spending Score Distribution", selected_theme),
                use_container_width=True,
            )

    if {"Annual_Income", "Spending_Score"}.issubset(clustered_df.columns):
        hover_cols = [
            col for col in ["Age", "Gender", "Cluster", "Segment_Name"]
            if col in clustered_df.columns
        ]

        st.plotly_chart(
            safe_scatter(
                clustered_df,
                x="Annual_Income",
                y="Spending_Score",
                color="Segment_Name",
                title="Annual Income vs Spending Score",
                theme=selected_theme,
                hover_data=hover_cols,
            ),
            use_container_width=True,
        )

    st.plotly_chart(
        plot_corr_heatmap(clustered_df, selected_features, selected_theme),
        use_container_width=True,
    )


# ============================================================
# TAB 4: CLUSTERING MODEL
# ============================================================

with tabs[3]:
    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("Selected K", str(selected_k))

    with c2:
        metric_card("Model Inertia", f"{model.inertia_:,.2f}")

    with c3:
        metric_card("Silhouette Score", f"{silhouette:.3f}" if not np.isnan(silhouette) else "N/A")

    st.markdown(
        """
        <div class="explain-box">
            <div class="section-title">Model Explanation</div>
            <p><strong>K-Means</strong> groups customers by distance similarity after feature scaling.</p>
            <p><strong>Inertia</strong> measures how compact the clusters are. Lower inertia usually means tighter clusters.</p>
            <p><strong>Silhouette Score</strong> measures cluster separation. A higher score usually means clearer clusters.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Clustered Customer Data")
    st.dataframe(clustered_df, use_container_width=True)

    st.markdown("### Elbow Method")
    inertia_values = []
    k_range = range(2, max_possible_k + 1)

    for k in k_range:
        temp_model = KMeans(n_clusters=k, random_state=42, n_init=10)
        temp_model.fit(scaled_features)
        inertia_values.append(temp_model.inertia_)

    elbow_df = pd.DataFrame({"K": list(k_range), "Inertia": inertia_values})

    fig_elbow = px.line(
        elbow_df,
        x="K",
        y="Inertia",
        markers=True,
        title="Elbow Method for Optimal K",
        template=get_plot_template(selected_theme),
    )
    fig_elbow.update_layout(height=450)
    st.plotly_chart(fig_elbow, use_container_width=True)


# ============================================================
# TAB 5: PCA / t-SNE
# ============================================================

with tabs[4]:
    st.markdown("### PCA Cluster Visualization")

    pca = PCA(n_components=2)
    pca_points = pca.fit_transform(scaled_features)

    pca_df = pd.DataFrame(
        {
            "PCA 1": pca_points[:, 0],
            "PCA 2": pca_points[:, 1],
            "Cluster": clustered_df["Cluster"].astype(str),
            "Segment": clustered_df["Segment_Name"],
        }
    )

    fig_pca = px.scatter(
        pca_df,
        x="PCA 1",
        y="PCA 2",
        color="Segment",
        symbol="Cluster",
        title="PCA Visualization of Customer Clusters",
        color_discrete_sequence=px.colors.qualitative.Bold,
        template=get_plot_template(selected_theme),
    )
    fig_pca.update_traces(marker=dict(size=10, opacity=0.85))
    fig_pca.update_layout(height=520)
    st.plotly_chart(fig_pca, use_container_width=True)

    st.markdown("### t-SNE Cluster Visualization")

    if len(clustered_df) > 10:
        perplexity = min(30, max(5, (len(clustered_df) - 1) // 3))

        if perplexity < len(clustered_df):
            tsne = TSNE(
                n_components=2,
                random_state=42,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
            )

            tsne_points = tsne.fit_transform(scaled_features)

            tsne_df = pd.DataFrame(
                {
                    "t-SNE 1": tsne_points[:, 0],
                    "t-SNE 2": tsne_points[:, 1],
                    "Cluster": clustered_df["Cluster"].astype(str),
                    "Segment": clustered_df["Segment_Name"],
                }
            )

            fig_tsne = px.scatter(
                tsne_df,
                x="t-SNE 1",
                y="t-SNE 2",
                color="Segment",
                symbol="Cluster",
                title="t-SNE Visualization of Customer Clusters",
                color_discrete_sequence=px.colors.qualitative.Bold,
                template=get_plot_template(selected_theme),
            )
            fig_tsne.update_traces(marker=dict(size=10, opacity=0.85))
            fig_tsne.update_layout(height=520)
            st.plotly_chart(fig_tsne, use_container_width=True)
        else:
            st.info("t-SNE skipped because the dataset is too small for the selected perplexity.")
    else:
        st.info("t-SNE requires more than 10 rows. Use a larger dataset to view t-SNE visualization.")


# ============================================================
# TAB 6: CLUSTER PROFILES
# ============================================================

with tabs[5]:
    st.markdown("### Cluster Profile Table")
    st.dataframe(profile_df, use_container_width=True)

    fig_profile = px.bar(
        profile_df,
        x="Cluster",
        y="Customer_Count",
        color="Segment_Name",
        text="Customer_Count",
        title="Customer Count per Cluster",
        color_discrete_sequence=px.colors.qualitative.Bold,
        template=get_plot_template(selected_theme),
    )
    fig_profile.update_layout(height=500)
    st.plotly_chart(fig_profile, use_container_width=True)

    if {"Annual_Income", "Spending_Score"}.issubset(profile_df.columns):
        fig_profile_scatter = px.scatter(
            profile_df,
            x="Annual_Income",
            y="Spending_Score",
            size="Customer_Count",
            color="Segment_Name",
            text="Cluster",
            title="Cluster Profile: Income vs Spending",
            color_discrete_sequence=px.colors.qualitative.Bold,
            template=get_plot_template(selected_theme),
        )
        fig_profile_scatter.update_layout(height=520)
        st.plotly_chart(fig_profile_scatter, use_container_width=True)


# ============================================================
# TAB 7: MARKETING STRATEGY
# ============================================================

with tabs[6]:
    st.markdown("### Marketing Strategy by Segment")

    for _, row in profile_df.iterrows():
        strategy = generate_marketing_strategy(row["Segment_Name"])

        st.markdown(
            f"""
            <div class="card">
                <div class="strategy-pill">Cluster {int(row['Cluster'])}</div>
                <div class="strategy-name">{row['Segment_Name']}</div>
                <p><strong>Customer behavior:</strong> {strategy['behavior']}</p>
                <p><strong>Business interpretation:</strong> {strategy['interpretation']}</p>
                <p><strong>Recommended marketing strategy:</strong> {strategy['strategy']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# TAB 8: DOWNLOADS
# ============================================================

with tabs[7]:
    st.markdown("### Download Project Outputs")

    c1, c2 = st.columns(2)

    with c1:
        st.download_button(
            label="Download Clustered Customers",
            data=dataframe_to_csv_bytes(clustered_df),
            file_name="clustered_customers.csv",
            mime="text/csv",
        )

    with c2:
        st.download_button(
            label="Download Cluster Profile",
            data=dataframe_to_csv_bytes(profile_df),
            file_name="cluster_profile.csv",
            mime="text/csv",
        )

    st.markdown(
        """
        <div class="card">
            <div class="section-title">Saved Files</div>
            <p class="small-note">
            The dashboard also saves output files automatically inside the <strong>outputs</strong> folder:
            <br>outputs/clustered_customers.csv
            <br>outputs/cluster_profile.csv
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
