import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# =======================================================
# PAGE COMPOSITION & CSS STYLING
# =======================================================
st.set_page_config(
    page_title="LTV Predictor Dashboard",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Applying custom CSS for light minimalism
st.markdown("""
<style>
    /* Global background layout */
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar Clean styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Sidebar text color override for Light Mode */
    [data-testid="stSidebar"] * {
        color: #334155 !important;
    }
    
    /* Input Fields */
    .stNumberInput input, .stSelectbox select, .stTextInput input, .stSlider > div > div > div > div > div {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        color: #1e293b !important;
        border-radius: 8px !important;
        transition: all 0.3s ease;
    }
    .stNumberInput input:focus, .stSelectbox select:focus, .stTextInput input:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2) !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #2563eb !important;
        color: white !important;
        border: none !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        transition: transform 0.3s ease, box-shadow 0.3s ease !important;
        width: 100%;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.1) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 20px -10px rgba(37, 99, 235, 0.4) !important;
    }

    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    div[data-testid="metric-container"] label {
        color: #64748b !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #0f172a !important;
    }
    
    /* Headers & Text */
    h1, h2, h3 {
        color: #0f172a !important;
        font-weight: 700;
        background: none !important;
        -webkit-text-fill-color: initial !important;
    }
    p, span, div {
        color: #1e293b;
    }
</style>
""", unsafe_allow_html=True)

# =======================================================
# CACHED DATA & MODEL LOADING
# =======================================================
@st.cache_data
def load_data():
    raw_path = "data/raw/customer_details.csv"
    processed_path = "data/processed/customer_details_processed.csv"
    
    # Try reading the data
    try:
        df_raw = pd.read_csv(raw_path)
        df_proc = pd.read_csv(processed_path)
    except FileNotFoundError:
        st.error("Data files not found! Make sure `data/processed/customer_details_processed.csv` exists.")
        return None, None
        
    # Re-calculate exact ltv formula as in processor.py to raw data if needed
    if 'ltv' not in df_proc.columns:
        df_proc['ltv'] = (0.30 * df_proc['Purchase Amount (USD)'] + 
                          0.25 * df_proc['Frequency of Purchases'] + 
                          0.20 * df_proc['Previous Purchases'] + 
                          0.15 * df_proc['Review Rating'] + 
                          0.10 * df_proc['Subscription Status'])
    
    return df_raw, df_proc

@st.cache_resource
def train_model(df):
    """Train XGBoost dynamically on the processed dataset features"""
    drop_cols = ["ltv", "customer_segment", "Purchase Amount (USD)", 
                 "Frequency of Purchases", "Previous Purchases", 
                 "Review Rating", "Subscription Status"]
    
    X = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    if 'ltv' not in df.columns:
        st.error('Target ltv variable missing from processed data.')
        st.stop()
        
    y = df["ltv"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_absolute_error(y_test, preds)) # Just a rough metric proxy for the dashboard
    r2 = r2_score(y_test, preds)
    
    return model, X.columns.tolist(), r2, rmse, X_test

# Load it fully
raw_df, proc_df = load_data()
if raw_df is None or proc_df is None:
    st.stop()

# Train Model on boot
with st.spinner("Initializing LTV Prediction Engine..."):
    model, feature_names, r2, rmse, X_test = train_model(proc_df)

# =======================================================
# SIDEBAR NAVIGATION
# =======================================================
st.sidebar.markdown("## ✨ **LTV Predictor**")
st.sidebar.markdown("Navigate the analytics engine.")
page = st.sidebar.radio("Navigation", ["Dashboard & EDA", "Predict Customer LTV"], label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.markdown(f"**Model R²**: `{r2:.3f}`")
st.sidebar.markdown(f"**Avg Target MAE**: `{rmse:.3f}`")

# =======================================================
# PAGE: DASHBOARD & EDA
# =======================================================
if page == "Dashboard & EDA":
    st.title("📊 Customer Cohort Analytics")
    st.markdown("Gain insights into purchasing behavior and predicted lifetime value across your customer base.")
    
    # KPIs Top Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{len(raw_df):,}")
    with col2:
        st.metric("Avg Purchase", f"${raw_df['Purchase Amount (USD)'].mean():.2f}")
    with col3:
        st.metric("Avg Review", f"{raw_df['Review Rating'].mean():.2f} ⭐")
    with col4:
        st.metric("Top Location", raw_df['Location'].mode()[0])

    st.divider()

    # Vizz Row 1
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Category Purchasing Volume")
        fig1 = px.histogram(raw_df, x="Category", color="Gender", barmode="group",
                            color_discrete_sequence=['#2563eb', '#14b8a6'])
        fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#1e293b")
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        st.subheader("Seasonality of Reviews")
        fig2 = px.box(raw_df, x="Season", y="Review Rating", color="Season",
                      color_discrete_sequence=['#2563eb', '#14b8a6', '#f59e0b', '#ef4444'])
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#1e293b")
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    
    # Feature Importance Plot
    st.subheader("Top Predictive Drivers of LTV")
    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:] # Top 10
    top_features = [feature_names[i] for i in indices]
    top_vals = importances[indices]
    
    fig3 = px.bar(x=top_vals, y=top_features, orientation='h',
                  color=top_vals, color_continuous_scale="Blues")
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
                       font_color="#1e293b", xaxis_title="Importance Weight", yaxis_title="Feature")
    st.plotly_chart(fig3, use_container_width=True)


# =======================================================
# PAGE: PREDICT LTV
# =======================================================
elif page == "Predict Customer LTV":
    st.title("🔮 Instant LTV Forecaster")
    st.markdown("Analyze existing customers interactively and forecast their potential Lifetime Value directly from the datastore.")
    
    st.markdown("### Select an Existing Customer")
    # Quick select from Raw DF since it corresponds index 1:1 with processed DF
    raw_df_with_idx = raw_df.copy()
    raw_df_with_idx['idx'] = raw_df_with_idx.index
    
    # We allow the user to select an existing customer to see their LTV exactly as it flowed via the pipe
    c1, c2 = st.columns([1, 2])
    with c1:
        selected_idx = st.selectbox(
            "Customer Selection (by sequential ID + Location)", 
            raw_df_with_idx['idx'].tolist(),
            format_func=lambda x: f"CUST-00{x} | ({raw_df_with_idx.loc[x, 'Age']}yrs, {raw_df_with_idx.loc[x, 'Location']})"
        )
    
    with c2:
        st.info("Since features must perfectly match standard scaling thresholds, the predictor selects live transformed feature rows.")

    st.divider()
    
    # Extract Real Raw Stats
    c_raw = raw_df.iloc[selected_idx]
    
    # True computed LTV baseline
    real_ltv = proc_df.iloc[selected_idx]['ltv']
    
    # Perform Inference
    # Using [[selected_idx]] returns a 1-row DataFrame, preserving dtypes correctly
    vector = proc_df[feature_names].iloc[[selected_idx]]
    pred_ltv = model.predict(vector)[0]
    
    # Layout 1: Customer Profile
    st.markdown("### Profile Matrix")
    stat1, stat2, stat3, stat4, stat5 = st.columns(5)
    stat1.metric("Age", c_raw['Age'])
    stat2.metric("Category", c_raw['Category'])
    stat3.metric("Gender", c_raw['Gender'])
    stat4.metric("Season", c_raw['Season'])
    stat5.metric("Location", c_raw['Location'])
    
    # Layout 2: Results Display
    st.markdown("---")
    st.markdown("### Deep Learning Forecast")
    
    res1, res2 = st.columns(2)
    with res1:
        st.markdown(f"""
        <div style="background: #eff6ff; border: 1px solid #2563eb; padding: 30px; border-radius: 12px; text-align: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">
            <h2 style="margin: 0; color: #1e293b; font-size: 1.5rem; background: none; -webkit-text-fill-color: #1e293b;">Predicted Multi-Factor LTV</h2>
            <h1 style="margin: 10px 0 0 0; font-size: 3.5rem; color: #2563eb; font-weight: 800; background: none; -webkit-text-fill-color: #2563eb;">{pred_ltv:.2f}</h1>
        </div>
        """, unsafe_allow_html=True)
        
    with res2:
        st.markdown(f"""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; padding: 30px; border-radius: 12px; text-align: center; height: 100%; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">
            <h2 style="margin: 0; color: #64748b; font-size: 1.5rem; background: none; -webkit-text-fill-color: #64748b;">Calculated Base LTV</h2>
            <h1 style="margin: 10px 0 0 0; font-size: 2.5rem; color: #334155; font-weight: 700; background: none; -webkit-text-fill-color: #334155;">{real_ltv:.2f}</h1>
            <p style="margin-top:20px; color: #94a3b8;">(Baseline target variable calculated previously)</p>
        </div>
        """, unsafe_allow_html=True)

    # Let the user interactively explore the underlying processed vector
    with st.expander("🔍 Peek at Scaled Feature Transform (Internal Vector)"):
        st.dataframe(vector)
