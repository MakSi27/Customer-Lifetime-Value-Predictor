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

# Applying custom CSS for dark glassmorphism
st.markdown("""
<style>
    /* Global background gradient */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.4) !important;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Input Fields */
    .stNumberInput input, .stSelectbox select, .stTextInput input, .stSlider > div > div > div > div > div {
        background: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #f8fafc !important;
        border-radius: 8px !important;
        transition: all 0.3s ease;
    }
    .stNumberInput input:focus, .stSelectbox select:focus, .stTextInput input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        transition: transform 0.3s ease, box-shadow 0.3s ease !important;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 20px -10px rgba(99, 102, 241, 0.6) !important;
    }

    /* Metric Cards */
    div[data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Headers & Text */
    h1, h2, h3 {
        background: linear-gradient(to right, #f8fafc, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
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
                            color_discrete_sequence=['#6366f1', '#e879f9'])
        fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        st.subheader("Seasonality of Reviews")
        fig2 = px.box(raw_df, x="Season", y="Review Rating", color="Season",
                      color_discrete_sequence=['#3b82f6', '#14b8a6', '#f59e0b', '#f43f5e'])
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    
    # Feature Importance Plot
    st.subheader("Top Predictive Drivers of LTV")
    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:] # Top 10
    top_features = [feature_names[i] for i in indices]
    top_vals = importances[indices]
    
    fig3 = px.bar(x=top_vals, y=top_features, orientation='h',
                  color=top_vals, color_continuous_scale="Purp")
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
                       font_color="#f8fafc", xaxis_title="Importance Weight", yaxis_title="Feature")
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
    # Extract Processed Feature Vector for Inference
    c_proc = proc_df.iloc[selected_idx][feature_names]
    
    # True computed LTV baseline
    real_ltv = proc_df.iloc[selected_idx]['ltv']
    
    # Perform Inference 
    vector = c_proc.to_frame().T
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
        <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid #6366f1; padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="margin: 0; color: #f8fafc; font-size: 1.5rem;">Predicted Multi-Factor LTV</h2>
            <h1 style="margin: 10px 0 0 0; font-size: 3.5rem; color: #818cf8; font-weight: 800;">{pred_ltv:.2f}</h1>
        </div>
        """, unsafe_allow_html=True)
        
    with res2:
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255,255,255,0.1); padding: 30px; border-radius: 12px; text-align: center; height: 100%;">
            <h2 style="margin: 0; color: #94a3b8; font-size: 1.5rem;">Calculated Base LTV</h2>
            <h1 style="margin: 10px 0 0 0; font-size: 2.5rem; color: #cbd5e1; font-weight: 700;">{real_ltv:.2f}</h1>
            <p style="margin-top:20px; color: #64748b;">(Baseline target variable calculated previously)</p>
        </div>
        """, unsafe_allow_html=True)

    # Let the user interactively explore the underlying processed vector
    with st.expander("🔍 Peek at Scaled Feature Transform (Internal Vector)"):
        st.dataframe(vector)
