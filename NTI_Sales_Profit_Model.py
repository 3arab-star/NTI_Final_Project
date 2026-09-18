# -*- coding: utf-8 -*-
"""
Sales & Profit Intelligence Dashboard
=====================================
Streamlit dashboard for sales analysis and profit prediction.

Model:
    RandomForestRegressor inside a scikit-learn Pipeline.

Required files:
    - trained_model.joblib
    - Sales Dataset cleaned with dashboard using Excel.xlsx

Place both files in the same folder as this app.py, or upload them
from the sidebar.
"""

import io
import os
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Global chart background (applies to every Plotly figure automatically)
# ---------------------------------------------------------------------
VISUAL_BG = "#152B26"
GRID_COLOR = "rgba(201, 241, 183, 0.14)"
TICK_COLOR = "#C9F1B7"
ACCENT_COLOR = "#C9F1B7"

pio.templates["brand"] = pio.templates["plotly_white"]
_brand_layout = pio.templates["brand"].layout
_brand_layout.paper_bgcolor = VISUAL_BG
_brand_layout.plot_bgcolor = VISUAL_BG
_brand_layout.font.color = "#FFFFFF"
_brand_layout.colorway = ["#C9F1B7", "#FFFFFF", "#8FCE7A", "#D9F2A8"]
_brand_layout.legend.font.color = "#FFFFFF"
_brand_layout.xaxis.gridcolor = GRID_COLOR
_brand_layout.yaxis.gridcolor = GRID_COLOR
_brand_layout.xaxis.tickfont.color = TICK_COLOR
_brand_layout.yaxis.tickfont.color = TICK_COLOR
_brand_layout.xaxis.title.font.color = "#FFFFFF"
_brand_layout.yaxis.title.font.color = "#FFFFFF"

pio.templates.default = "brand"

# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Sales & Profit Intelligence",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Logo (SVG-free PNG with transparent background). Shown in the sidebar
# and next to the app header via st.logo.
LOGO_PATH = os.path.join(
    APP_DIR, "visionflow-background-removed-1789748614906.png"
)
if os.path.isfile(LOGO_PATH):
    st.logo(LOGO_PATH, size="large")

# IMPORTANT:
# Do not put an absolute Windows path inside os.path.join().
# These files are expected to be next to app.py.
DEFAULT_MODEL_PATH = os.path.join(APP_DIR, "trained_model.joblib")
DEFAULT_DATA_PATH = os.path.join(
    APP_DIR, "Sales Dataset cleaned with dashboard using Excel.xlsx"
)

NUMERIC_FEATURES = ["Sales", "Discount", "Quantity", "Cost Of Items"]
CATEGORICAL_FEATURES = ["Category", "Sub-Category", "Segment"]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# ---------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------
CUSTOM_CSS = """
<style>
html, body, [class*="css"] {
    font-family: "Segoe UI", Arial, sans-serif;
}

.stApp {
    background-color: #152B26;
    color: #FFFFFF;
}

div[data-testid="stMetricValue"] { color: #FFFFFF; }
div[data-testid="stMetricLabel"] { color: #C9F1B7; }

section[data-testid="stSidebar"] button {
    justify-content: flex-start;
    font-weight: 600;
}

section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] {
    background: transparent !important;
    border: 1px solid rgba(201, 241, 183, 0.28) !important;
    color: #C9F1B7 !important;
}

section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]:hover {
    background: rgba(201, 241, 183, 0.10) !important;
    border-color: #C9F1B7 !important;
    color: #FFFFFF !important;
}

section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {
    background: #C9F1B7 !important;
    border: none !important;
    color: #152B26 !important;
}

section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"]:hover {
    background: #E4F7EC !important;
    color: #152B26 !important;
}

button[data-testid="stBaseButton-primary"] {
    color: #152B26 !important;
    font-weight: 700;
}

button[data-testid="stBaseButton-primary"]:hover {
    color: #152B26 !important;
}

.main-header {
    padding: 1.6rem 2rem;
    border-radius: 18px;
    background: linear-gradient(120deg, #0A1D18 0%, #152B26 50%, #1E4539 100%);
    border: 1px solid rgba(201, 241, 183, 0.22);
    color: #FFFFFF;
    margin-bottom: 1.4rem;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
}

.main-header h1 {
    margin: 0;
    font-size: 2rem;
    font-weight: 800;
    color: #FFFFFF;
}

.main-header p {
    margin: 0.3rem 0 0 0;
    opacity: 0.85;
    font-size: 1rem;
    color: #C9F1B7;
}

.kpi-card {
    background: #1C3B34;
    border-radius: 16px;
    padding: 1.1rem 0.8rem;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.28);
    border-left: 5px solid #C9F1B7;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 140px;
    width: 100%;
    box-sizing: border-box;
}

.kpi-card h3 {
    margin: 0;
    font-size: 0.85rem;
    color: #C9F1B7;
    font-weight: 600;
    text-align: center;
    width: 100%;
}

.kpi-card p {
    margin: 0.35rem 0 0 0;
    font-size: 1.3rem;
    font-weight: 800;
    color: #FFFFFF;
    white-space: nowrap;
    text-align: center;
}

.section-title {
    font-size: 1.3rem;
    font-weight: 800;
    margin: 1.2rem 0 0.6rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 3px solid #C9F1B7;
    display: inline-block;
    color: #FFFFFF;
}

footer {
    visibility: hidden;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(
    '<link rel="stylesheet" '
    'href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained model...")
def load_model(model_source):
    """
    Load the trained model.

    The compatibility patch below is kept for older scikit-learn models
    that may contain the private _RemainderColsList object.
    """
    # Compatibility patch for some old scikit-learn models.
    try:
        import sklearn.compose._column_transformer as ct

        if not hasattr(ct, "_RemainderColsList"):
            class _RemainderColsList(list):
                pass

            ct._RemainderColsList = _RemainderColsList
    except Exception:
        pass

    if isinstance(model_source, (str, os.PathLike)):
        return joblib.load(model_source)

    return joblib.load(model_source)


# ---------------------------------------------------------------------
# Data loading and cleaning
# ---------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and cleaning sales data...")
def load_and_clean_data(data_source):
    sheets = pd.read_excel(data_source, sheet_name=None)

    required_sheets = {"Return", "People", "Orders"}
    missing_sheets = required_sheets.difference(sheets.keys())
    if missing_sheets:
        raise ValueError(
            f"Missing required Excel sheets: {', '.join(sorted(missing_sheets))}"
        )

    df_return = sheets["Return"].copy()
    df_people = sheets["People"].copy()
    df = sheets["Orders"].copy()

    # Remove completely empty columns.
    df = df.dropna(axis=1, how="all")

    columns_to_remove = [
        "Unnamed: 46",
        "Unnamed: 42",
        "Unnamed: 43",
        "Column24",
        "Column25",
        "Column23",
        "Column22",
        "Column21",
    ]
    df = df.drop(columns=columns_to_remove, errors="ignore")

    # Remove rows containing missing values, matching the original workflow.
    df = df.dropna().reset_index(drop=True)
    df = df.drop(columns=["Segment2"], errors="ignore")

    # Clean text columns.
    text_cols = df.select_dtypes(include="object").columns
    for col in text_cols:
        df[col] = df[col].astype("string").str.strip()

    # Fix known column names.
    df = df.rename(
        columns={
            "Coast Of Items": "Cost Of Items",
            "Ship Date.1": "Ship Date",
        }
    )

    if "Quantity" in df.columns:
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
        df = df.dropna(subset=["Quantity"])
        df["Quantity"] = df["Quantity"].astype(int)

    if "Postal Code" in df.columns:
        df["Postal Code"] = (
            df["Postal Code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
        )

    # Process returns.
    if "Returned" in df_return.columns:
        df_return["Returned"] = df_return["Returned"].map({"Yes": 1, "No": 0})
        df_return["Returned"] = df_return["Returned"].fillna(0).astype(int)

    orders_with_return = pd.merge(
        df,
        df_return,
        on="Order ID",
        how="left",
    )

    orders_with_return["Returned"] = (
        orders_with_return["Returned"].fillna(0).astype(int)
    )

    # Add sales representative / people information.
    orders = pd.merge(
        orders_with_return,
        df_people,
        on="Region",
        how="left",
    )

    keep_cols = [
        "Order ID",
        "Returned",
        "Customer ID",
        "Segment",
        "Customer Name",
        "City",
        "State",
        "Order Date",
        "Product Name",
        "Category",
        "Sub-Category",
        "Region",
        "Person",
        "Price Of Items Before dis",
        "Sales Before Discount",
        "Sales",
        "Discount",
        "Quantity",
        "Price Of Item After Dis",
        "Discount Value",
        "Cost Of Items",
        "Profit",
    ]

    available_cols = [c for c in keep_cols if c in orders.columns]
    orders = orders[available_cols].copy()

    # Clean string columns.
    str_cols = orders.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        orders[col] = orders[col].astype("string").str.strip()

    # Preserve the original capitalization behavior.
    for col in str_cols:
        orders[col] = orders[col].str.capitalize()

    # Dates and numeric fields.
    if "Order Date" in orders.columns:
        orders["Order Date"] = pd.to_datetime(
            orders["Order Date"], errors="coerce"
        )

    numeric_cols = [
        "Sales",
        "Discount",
        "Quantity",
        "Cost Of Items",
        "Profit",
    ]
    for col in numeric_cols:
        if col in orders.columns:
            orders[col] = pd.to_numeric(orders[col], errors="coerce")

    orders = orders.dropna(
        subset=[c for c in MODEL_FEATURES + ["Profit"] if c in orders.columns]
    ).reset_index(drop=True)

    if "Profit Margin %" not in orders.columns:
        orders["Profit Margin %"] = np.where(
            orders["Sales"] != 0,
            (orders["Profit"] / orders["Sales"]) * 100,
            0,
        )

    return orders


# ---------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_test_metrics(_model, orders_df):
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )
    from sklearn.model_selection import train_test_split

    missing_features = [c for c in MODEL_FEATURES if c not in orders_df.columns]
    if missing_features:
        raise ValueError(
            f"Missing model features in the dataset: {', '.join(missing_features)}"
        )

    X = orders_df[MODEL_FEATURES].copy()
    y = orders_df["Profit"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # X_train and y_train are intentionally created to reproduce the
    # same 80/20 evaluation split used in the original code.
    _ = X_train, y_train

    y_pred = _model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "n_test": len(y_test),
    }


# ---------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_feature_importance(_model):
    if not hasattr(_model, "named_steps"):
        return None

    regressor = _model.named_steps.get("regressor")

    # Support both possible spellings.
    preproc = (
        _model.named_steps.get("preprocessor")
        or _model.named_steps.get("preproccessor")
    )

    if (
        regressor is None
        or preproc is None
        or not hasattr(regressor, "feature_importances_")
    ):
        return None

    importances = regressor.feature_importances_

    try:
        feat_names = list(preproc.get_feature_names_out())
    except Exception:
        feat_names = [
            f"feature_{i}" for i in range(len(importances))
        ]

    grouped = {}

    for name, importance in zip(feat_names, importances):
        clean = name.split("__", 1)[-1]
        base = clean

        for cat in CATEGORICAL_FEATURES:
            if clean.startswith(cat + "_"):
                base = cat
                break

        grouped[base] = grouped.get(base, 0) + importance

    imp_df = pd.DataFrame(
        {
            "Feature": list(grouped.keys()),
            "Importance": list(grouped.values()),
        }
    )

    return imp_df.sort_values(
        "Importance",
        ascending=False,
    ).reset_index(drop=True)


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def fmt_money(value):
    return f"${value:,.0f}"


def kpi_card(col, label, value):
    with col:
        st.markdown(
            f"""
            <div class="kpi-card">
                <h3>{label}</h3>
                <p>{value}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


_ALERT_STYLES = {
    "success": ("fa-circle-check", "#C9F1B7", "#1C3B34"),
    "warning": ("fa-triangle-exclamation", "#F2C879", "#1C3B34"),
    "error": ("fa-circle-xmark", "#F2A69B", "#1C3B34"),
    "info": ("fa-circle-info", "#C9F1B7", "#1C3B34"),
}


def alert(text, kind="info"):
    """Styled status message using a Font Awesome icon instead of an emoji."""
    icon, color, bg = _ALERT_STYLES.get(kind, _ALERT_STYLES["info"])
    st.markdown(
        f"""
        <div style="background:{bg}; border-left:4px solid {color}; border-radius:8px;
                     padding:0.55rem 0.85rem; margin:0.35rem 0; color:{color} !important;
                     font-weight:600; font-size:0.92rem;">
            <i class="fa-solid {icon}" style="margin-right:0.5rem; color:{color} !important;"></i>{text}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
NAV_ITEMS = [
    ("Home", "home"),
    ("Analytics Dashboard", "bar_chart"),
    ("Profit Prediction", "smart_toy"),
    ("Data Explorer", "search"),
    ("Model Performance", "psychology"),
]

if "page" not in st.session_state:
    st.session_state.page = "Home"

with st.sidebar:
    st.markdown(
        '<div style="font-weight:800; font-size:1rem; margin-bottom:0.6rem;">'
        '<i class="fa-solid fa-compass"></i> Navigation</div>',
        unsafe_allow_html=True,
    )

    for label, icon in NAV_ITEMS:
        is_active = st.session_state.page == label
        if st.button(
            label,
            key=f"nav_{label}",
            icon=f":material/{icon}:",
            width="stretch",
            type="primary" if is_active else "secondary",
        ):
            st.session_state.page = label

    page = st.session_state.page

    st.markdown("---")

    model_file_obj = None
    data_file_obj = None

    with st.expander("Settings", icon=":material/settings:", expanded=False):
        # Model
        if os.path.isfile(DEFAULT_MODEL_PATH):
            model_source = DEFAULT_MODEL_PATH
            alert("Trained model found automatically.", "success")
        else:
            alert("Upload the trained model.", "warning")
            model_file_obj = st.file_uploader(
                "Trained model",
                type=["joblib", "pkl"],
            )
            model_source = model_file_obj

        # Dataset
        if os.path.isfile(DEFAULT_DATA_PATH):
            data_source = DEFAULT_DATA_PATH
            alert("Excel dataset found automatically.", "success")
        else:
            alert("Upload the sales dataset.", "warning")
            data_file_obj = st.file_uploader(
                "Sales dataset",
                type=["xlsx"],
            )
            data_source = data_file_obj

    st.markdown("---")
    st.caption("Built with Streamlit + scikit-learn + Plotly")


# ---------------------------------------------------------------------
# Validate files
# ---------------------------------------------------------------------
if model_source is None or data_source is None:
    st.markdown(
        """
        <div class="main-header">
            <h1><i class="fa-solid fa-chart-column"></i> Sales &amp; Profit Intelligence</h1>
            <p>
                Provide the trained model and sales Excel dataset from
                the sidebar to start.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    alert(
        "Open <strong>Settings</strong> in the sidebar to provide both required "
        "files: the trained model and the Excel dataset.",
        "info",
    )
    st.stop()


# ---------------------------------------------------------------------
# Load model and data
# ---------------------------------------------------------------------
try:
    model = load_model(model_source)
except Exception as e:
    alert(f"Failed to load the trained model: {e}", "error")
    st.stop()

try:
    orders = load_and_clean_data(data_source)
except Exception as e:
    alert(f"Failed to load or clean the sales data: {e}", "error")
    st.stop()


# ---------------------------------------------------------------------
# Home page
# ---------------------------------------------------------------------
if page == "Home":
    st.markdown(
        """
        <div class="main-header">
            <h1><i class="fa-solid fa-chart-column"></i> Sales &amp; Profit Intelligence Dashboard</h1>
            <p>
                Comprehensive sales analysis, data exploration,
                and AI-powered profit prediction.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    kpi_card(c1, "Total Sales", fmt_money(orders["Sales"].sum()))
    kpi_card(c2, "Total Profit", fmt_money(orders["Profit"].sum()))
    kpi_card(c3, "Number of Orders", f"{orders['Order ID'].nunique():,}")
    kpi_card(c4, "Average Profit Margin", f"{orders['Profit Margin %'].mean():.1f}%")
    kpi_card(c5, "Return Rate", f"{orders['Returned'].mean() * 100:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown(
            '<div class="section-title"><i class="fa-solid fa-chart-line"></i> Sales &amp; Profit Trend Over Time</div>',
            unsafe_allow_html=True,
        )

        trend = (
            orders.dropna(subset=["Order Date"])
            .groupby(orders["Order Date"].dt.to_period("M"))[["Sales", "Profit"]]
            .sum()
            .reset_index()
        )

        trend["Order Date"] = trend["Order Date"].dt.to_timestamp()

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=trend["Order Date"],
                y=trend["Sales"],
                name="Sales",
                mode="lines",
                line=dict(color="#C9F1B7", width=3),
                fill="tozeroy",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=trend["Order Date"],
                y=trend["Profit"],
                name="Profit",
                mode="lines",
                line=dict(color="#FFFFFF", width=3),
            )
        )

        fig.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", y=1.1),
            hovermode="x unified",
        )

        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown(
            '<div class="section-title"><i class="fa-solid fa-tag"></i> Profit by Segment</div>',
            unsafe_allow_html=True,
        )

        seg = (
            orders.groupby("Segment")[["Sales", "Profit"]]
            .sum()
            .reset_index()
        )

        fig2 = px.pie(
            seg,
            names="Segment",
            values="Profit",
            hole=0.45,
            color_discrete_sequence=["#C9F1B7", "#FFFFFF", "#8FCE7A"],
        )

        fig2.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=30, b=10),
        )

        st.plotly_chart(fig2, width="stretch")

    st.markdown(
        '<div class="section-title"><i class="fa-solid fa-folder-open"></i> Project Overview</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        - **Data:** Sales, returns, and people data after cleaning and preprocessing.
        - **Model:** `RandomForestRegressor` inside a `Pipeline` with
          `ColumnTransformer` and One-Hot Encoding for categorical features.
        - **Target:** Predicting `Profit` from sales, discount, quantity,
          item cost, category, sub-category, and segment.
        - Use the sidebar to navigate between analytics, prediction,
          data exploration, and model performance.
        """
    )


# ---------------------------------------------------------------------
# Analytics Dashboard
# ---------------------------------------------------------------------
elif page == "Analytics Dashboard":
    st.markdown(
        """
        <div class="main-header">
            <h1><i class="fa-solid fa-chart-column"></i> Sales &amp; Profit Analytics</h1>
            <p>Filter the data and explore the results interactively.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Filters", icon=":material/filter_alt:", expanded=True):
        f1, f2, f3, f4 = st.columns(4)

        categories = f1.multiselect(
            "Category",
            sorted(orders["Category"].dropna().unique()),
        )

        regions = f2.multiselect(
            "Region",
            sorted(orders["Region"].dropna().unique()),
        )

        segments = f3.multiselect(
            "Segment",
            sorted(orders["Segment"].dropna().unique()),
        )

        min_date = orders["Order Date"].min()
        max_date = orders["Order Date"].max()

        date_range = f4.date_input(
            "Date Range",
            (min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
        )

    filtered = orders.copy()

    if categories:
        filtered = filtered[filtered["Category"].isin(categories)]

    if regions:
        filtered = filtered[filtered["Region"].isin(regions)]

    if segments:
        filtered = filtered[filtered["Segment"].isin(segments)]

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start = pd.Timestamp(date_range[0])
        # Add one day so the selected end date is fully included.
        end = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)

        filtered = filtered[
            (filtered["Order Date"] >= start)
            & (filtered["Order Date"] < end)
        ]

    if filtered.empty:
        alert("No data matches the selected filters.", "warning")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)

    kpi_card(c1, "Total Sales", fmt_money(filtered["Sales"].sum()))
    kpi_card(c2, "Total Profit", fmt_money(filtered["Profit"].sum()))
    kpi_card(c3, "Number of Orders", f"{filtered['Order ID'].nunique():,}")
    kpi_card(c4, "Average Discount", f"{filtered['Discount'].mean() * 100:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="section-title"><i class="fa-solid fa-box"></i> Sales &amp; Profit by Category</div>',
            unsafe_allow_html=True,
        )

        cat_sum = (
            filtered.groupby("Category")[["Sales", "Profit"]]
            .sum()
            .reset_index()
        )

        fig = px.bar(
            cat_sum,
            x="Category",
            y=["Sales", "Profit"],
            barmode="group",
            color_discrete_sequence=["#C9F1B7", "#FFFFFF"],
        )

        fig.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=20, b=10),
            legend_title_text="",
        )

        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown(
            '<div class="section-title"><i class="fa-solid fa-puzzle-piece"></i> Profit by Sub-Category</div>',
            unsafe_allow_html=True,
        )

        sub_sum = (
            filtered.groupby("Sub-Category")["Profit"]
            .sum()
            .reset_index()
            .sort_values("Profit")
        )

        fig = px.bar(
            sub_sum,
            x="Profit",
            y="Sub-Category",
            orientation="h",
            color="Profit",
            color_continuous_scale=[[0, "#2A5A4C"], [1, "#C9F1B7"]],
        )

        fig.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=20, b=10),
            coloraxis_showscale=False,
        )

        st.plotly_chart(fig, width="stretch")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown(
            '<div class="section-title"><i class="fa-solid fa-earth-americas"></i> Top 10 States by Profit</div>',
            unsafe_allow_html=True,
        )

        top_states = (
            filtered.groupby("State")["Profit"]
            .sum()
            .reset_index()
            .sort_values("Profit", ascending=False)
            .head(10)
        )

        fig = px.bar(
            top_states,
            x="Profit",
            y="State",
            orientation="h",
            color="Profit",
            color_continuous_scale=[[0, "#2A5A4C"], [1, "#C9F1B7"]],
        )

        fig.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=20, b=10),
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
        )

        st.plotly_chart(fig, width="stretch")

    with col4:
        st.markdown(
            '<div class="section-title"><i class="fa-solid fa-link"></i> Numeric Feature Correlation</div>',
            unsafe_allow_html=True,
        )

        num_cols = filtered.select_dtypes(include="number").columns
        corr = filtered[num_cols].corr()

        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale=[[0, "#0E201C"], [0.5, "#3A6B5A"], [1, "#C9F1B7"]],
            aspect="auto",
        )

        fig.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=20, b=10),
        )

        st.plotly_chart(fig, width="stretch")

    st.markdown(
        '<div class="section-title"><i class="fa-solid fa-rotate-left"></i> Return Rate</div>',
        unsafe_allow_html=True,
    )

    ret_summary = (
        filtered["Returned"]
        .map({1: "Returned", 0: "Not Returned"})
        .value_counts()
        .reset_index()
    )

    ret_summary.columns = ["Status", "Count"]

    fig = px.pie(
        ret_summary,
        names="Status",
        values="Count",
        hole=0.5,
        color_discrete_sequence=["#C9F1B7", "#55846E"],
    )

    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=20, b=10),
    )

    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------
# Profit Prediction
# ---------------------------------------------------------------------
elif page == "Profit Prediction":
    st.markdown(
        """
        <div class="main-header">
            <h1><i class="fa-solid fa-robot"></i> Profit Prediction</h1>
            <p>Enter order information and predict the expected profit.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cat_options = sorted(
        orders["Category"].dropna().unique().tolist()
    )
    sub_options = sorted(
        orders["Sub-Category"].dropna().unique().tolist()
    )
    seg_options = sorted(
        orders["Segment"].dropna().unique().tolist()
    )

    left, right = st.columns([1, 1.2])

    with left:
        st.markdown(
            '<div class="section-title"><i class="fa-solid fa-pen-to-square"></i> Order Information</div>',
            unsafe_allow_html=True,
        )

        category = st.selectbox("Category", cat_options)
        sub_category = st.selectbox("Sub-Category", sub_options)
        segment = st.selectbox("Segment", seg_options)

        sales = st.number_input(
            "Sales",
            min_value=0.0,
            value=250.0,
            step=10.0,
        )

        cost = st.number_input(
            "Cost Of Items",
            min_value=0.0,
            value=150.0,
            step=10.0,
        )

        quantity = st.number_input(
            "Quantity",
            min_value=1,
            value=3,
            step=1,
        )

        discount = st.slider(
            "Discount",
            min_value=0.0,
            max_value=0.8,
            value=0.2,
            step=0.05,
        )

        predict_clicked = st.button(
            "Predict Profit",
            icon=":material/auto_awesome:",
            width="stretch",
            type="primary",
        )

    input_row = pd.DataFrame(
        [
            {
                "Category": category,
                "Sub-Category": sub_category,
                "Segment": segment,
                "Sales": sales,
                "Discount": discount,
                "Quantity": quantity,
                "Cost Of Items": cost,
            }
        ]
    )

    with right:
        st.markdown(
            '<div class="section-title"><i class="fa-solid fa-thumbtack"></i> Prediction Result</div>',
            unsafe_allow_html=True,
        )

        # Do not predict automatically on every Streamlit rerun.
        if predict_clicked:
            try:
                predicted_profit = float(
                    model.predict(input_row[MODEL_FEATURES])[0]
                )
            except Exception as e:
                alert(f"Prediction failed: {e}", "error")
                predicted_profit = None

            if predicted_profit is not None:
                margin = (
                    predicted_profit / sales * 100
                    if sales
                    else 0
                )

                m1, m2, m3 = st.columns(3)

                m1.metric(
                    "Predicted Profit",
                    f"${predicted_profit:,.2f}",
                )

                m2.metric(
                    "Profit Margin",
                    f"{margin:.1f}%",
                )

                m3.metric(
                    "Total Item Cost",
                    f"${cost * quantity:,.2f}",
                )

                if predicted_profit > 0:
                    status_icon, status_color = "fa-circle-check", "#C9F1B7"
                    status_text = "Positive Profit"
                else:
                    status_icon, status_color = "fa-circle-exclamation", "#F2C879"
                    status_text = "Expected Loss"

                st.markdown(
                    f"""
                    <div style="background:#1C3B34; border-left:4px solid {status_color};
                                 border-radius:8px; padding:0.55rem 0.85rem; margin:0.35rem 0;
                                 color:{status_color}; font-weight:700; font-size:0.95rem;">
                        <i class="fa-solid {status_icon}" style="margin-right:0.5rem;"></i>
                        Status: {status_text}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    '<div class="section-title"><i class="fa-solid fa-arrow-trend-down"></i> Discount Impact on Predicted Profit</div>',
                    unsafe_allow_html=True,
                )

                discounts_range = np.round(
                    np.arange(0.0, 0.85, 0.05),
                    2,
                )

                scenario_rows = pd.DataFrame(
                    [
                        {
                            "Category": category,
                            "Sub-Category": sub_category,
                            "Segment": segment,
                            "Sales": sales,
                            "Discount": d,
                            "Quantity": quantity,
                            "Cost Of Items": cost,
                        }
                        for d in discounts_range
                    ]
                )

                scenario_rows["Predicted Profit"] = model.predict(
                    scenario_rows[MODEL_FEATURES]
                )

                fig = px.line(
                    scenario_rows,
                    x="Discount",
                    y="Predicted Profit",
                    markers=True,
                    color_discrete_sequence=["#C9F1B7"],
                )

                fig.add_vline(
                    x=discount,
                    line_dash="dash",
                    line_color="#FFFFFF",
                    annotation_text="Current Discount",
                )

                fig.update_layout(
                    height=340,
                    margin=dict(l=10, r=10, t=20, b=10),
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                )
        else:
            alert("Enter the order information and click <strong>Predict Profit</strong>.", "info")

    st.markdown(
        '<div class="section-title"><i class="fa-solid fa-chart-column"></i> Comparison with Similar Historical Data</div>',
        unsafe_allow_html=True,
    )

    similar = orders[
        (orders["Category"] == category)
        & (orders["Sub-Category"] == sub_category)
    ]

    if not similar.empty:
        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Average Historical Profit",
            f"${similar['Profit'].mean():,.2f}",
        )

        c2.metric(
            "Average Historical Sales",
            f"${similar['Sales'].mean():,.2f}",
        )

        c3.metric(
            "Similar Historical Orders",
            f"{len(similar):,}",
        )
    else:
        st.caption(
            "No historical records were found for this category/sub-category combination."
        )


# ---------------------------------------------------------------------
# Data Explorer
# ---------------------------------------------------------------------
elif page == "Data Explorer":
    st.markdown(
        """
        <div class="main-header">
            <h1><i class="fa-solid fa-magnifying-glass"></i> Data Explorer</h1>
            <p>Search, inspect, and download the cleaned dataset.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    search = st.text_input(
        "Quick Search (customer, product, city, etc.)",
        "",
    )

    view = orders.copy()

    if search:
        mask = (
            view.astype(str)
            .apply(
                lambda col: col.str.contains(
                    search,
                    case=False,
                    na=False,
                    regex=False,
                )
            )
            .any(axis=1)
        )
        view = view[mask]

    st.dataframe(
        view,
        width="stretch",
        height=520,
    )

    st.caption(
        f"Displayed rows: {len(view):,} of {len(orders):,}"
    )

    csv_buffer = io.StringIO()
    view.to_csv(csv_buffer, index=False)

    st.download_button(
        "Download Displayed Data as CSV",
        icon=":material/download:",
        data=csv_buffer.getvalue(),
        file_name=f"sales_data_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------
# Model Performance
# ---------------------------------------------------------------------
elif page == "Model Performance":
    st.markdown(
        """
        <div class="main-header">
            <h1><i class="fa-solid fa-brain"></i> Model Performance</h1>
            <p>Evaluate the model and inspect the most important features.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        metrics = compute_test_metrics(model, orders)

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "MAE",
            f"{metrics['mae']:.2f}",
        )

        c2.metric(
            "RMSE",
            f"{metrics['rmse']:.2f}",
        )

        c3.metric(
            "R² Score",
            f"{metrics['r2'] * 100:.2f}%",
        )

        c4.metric(
            "Test Set Size",
            f"{metrics['n_test']:,}",
        )

        st.caption(
            "R² is displayed as a percentage here. For example, 92.79% "
            "corresponds to an R² value of 0.9279."
        )

    except Exception as e:
        alert(f"Unable to calculate model performance automatically: {e}", "warning")

    st.markdown(
        '<div class="section-title"><i class="fa-solid fa-star"></i> Most Important Features</div>',
        unsafe_allow_html=True,
    )

    imp_df = compute_feature_importance(model)

    if imp_df is not None:
        fig = px.bar(
            imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=[[0, "#2A5A4C"], [1, "#C9F1B7"]],
        )

        fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=20, b=10),
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )
    else:
        alert("Feature importance is not available for this model.", "info")

    st.markdown(
        '<div class="section-title"><i class="fa-solid fa-gear"></i> Pipeline Details</div>',
        unsafe_allow_html=True,
    )

    st.code(
        str(model),
        language="text",
    )

    st.markdown(
        """
        **Model notes:**

        - Model: `RandomForestRegressor` inside a `sklearn.Pipeline`.
        - Numeric features: `Sales`, `Discount`, `Quantity`, `Cost Of Items`.
        - Categorical features: `Category`, `Sub-Category`, `Segment`.
        - Categorical features are processed using One-Hot Encoding.
        - R² is a regression metric; a value closer to 1 indicates that
          the model explains more of the variation in the target.
        """
    )
