import pandas as pd
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from scipy.stats import zscore
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="NeuroPulse++ | Epilepsy AI Dashboard", layout="wide")

# --- CUSTOM STYLES ---
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #2E0F15 !important;
        color: #FAFFF0 !important;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: #842C35 !important;
        backdrop-filter: blur(6px);
        color: #FAFFF0 !important;
        border-right: 1px solid #EB5456;
    }

    /* Headings */
    h1, h2, h3 {
        color: #FF9FA2;
    }

    /* Metrics */
    [data-testid="metric-container"] {
        background-color: #EB5456;
        color: #FAFFF0;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 0 10px rgba(216, 76, 76, 0.3);
    }

    /* Tabs */
    .stTabs [role="tablist"] {
        background: #842C35;
        border-radius: 10px;
    }
    .stTabs [role="tab"] {
        color: #FAFFF0;
        padding: 10px;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 4px solid #FF6968;
        color: #FF9FA2;
    }

    /* Buttons */
    .stButton>button {
        background-color: #D84C4C;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #B43E3E;
        transition: 0.3s ease-in-out;
    }

    /* Inputs */
    .css-1cpxqw2 {
        background-color: #3A1D20 !important;
        border: 1px solid #EB5456 !important;
        color: white !important;
    }

    </style>
""", unsafe_allow_html=True)


# --- HEADER ---
st.markdown("<h1>🧠 NeuroPulse++</h1>", unsafe_allow_html=True)
st.markdown("AI-powered dashboard for epilepsy DALY prediction, clustering, and human insight.")
st.caption("📊 Source: WHO Global Health Estimates (2019)")
st.markdown("---")

# --- SIDEBAR CONTROLS ---
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("""
<div style='font-size:16px; font-weight:bold; color:#FF9FA2; padding: 6px 0;'>
🎨 Theme: Premium Coral Red
</div>
""", unsafe_allow_html=True)

show_male = st.sidebar.checkbox("👨 Show Male", True)
show_female = st.sidebar.checkbox("👩 Show Female", True)
show_prediction = st.sidebar.checkbox("🔮 Show 2025 Prediction", True)
show_cluster = st.sidebar.checkbox("🧬 Cluster Analysis", False)
chart_style = st.sidebar.radio("📈 Chart Style", ["Line", "Bar"])

# 👇 Move this here after df is loaded
selected_group = st.sidebar.multiselect(
    "🔎 Filter by Age Group",
    options=sorted(df["Age Group"].unique().tolist())
)

# --- DATA LOAD & PREP ---
df = pd.read_csv("epilepsy_daly_by_age_2019.csv")
df["DALY Male 2015"] = df["DALY Male"] * 1.05
df["DALY Female 2015"] = df["DALY Female"] * 1.08
age_groups = df["Age Group"]

# --- Apply Filters ---
if selected_group:
    df = df[df["Age Group"].isin(selected_group)]

# --- PREDICTION ---
def predict_daly(gender):
    preds = []
    for i in range(len(df)):
        X = np.array([[2015], [2019]])
        y = np.array([df.loc[i, f'DALY {gender} 2015'], df.loc[i, f'DALY {gender}']])
        model = LinearRegression().fit(X, y)
        preds.append(model.predict([[2025]])[0])
    df[f'DALY {gender} 2025'] = preds

if show_prediction:
    if show_male: predict_daly("Male")
    if show_female: predict_daly("Female")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🧠 ML Insights", "📦 Data & Tools"])

# --- TAB 1: MAIN DASHBOARD ---
with tab1:
    st.subheader("📌 Key Metrics")
    col1, col2, col3 = st.columns(3)
    if show_male:
        peak = df["DALY Male"].idxmax()
        col1.metric("👨 Peak Male DALY", f"{df['DALY Male'].max():,.0f}", df.loc[peak, "Age Group"])
    if show_female:
        peak = df["DALY Female"].idxmax()
        col2.metric("👩 Peak Female DALY", f"{df['DALY Female'].max():,.0f}", df.loc[peak, "Age Group"])
    if show_prediction:
        pred_avg = df[[c for c in df.columns if '2025' in c]].mean().mean()
        col3.metric("🔮 Avg Predicted 2025", f"{pred_avg:,.0f}")

    # --- CHART ---
    st.markdown("### 📈 DALY by Age Group")
    fig = go.Figure()

    def add_trace(label, y_data, color, dash="solid"):
        mode = "lines+markers" if chart_style == "Line" else "markers"
        fig.add_trace(go.Scatter(x=age_groups, y=y_data, mode=mode,
                                 name=label, line=dict(color=color, dash=dash)))

    if show_male: add_trace("Male 2019", df["DALY Male"], "blue")
    if show_female: add_trace("Female 2019", df["DALY Female"], "green")
    if show_prediction:
        if show_male and "DALY Male 2025" in df: add_trace("Male 2025", df["DALY Male 2025"], "orange", "dash")
        if show_female and "DALY Female 2025" in df: add_trace("Female 2025", df["DALY Female 2025"], "red", "dash")

    fig.update_layout(title="DALY by Age Group", hovermode="x unified", height=500,
                      xaxis_title="Age Group", yaxis_title="DALY",
                      legend=dict(orientation="h"))
    fig.update_traces(marker=dict(size=6), selector=dict(mode='lines+markers'))
    fig.update_layout(transition_duration=500)
    st.plotly_chart(fig, use_container_width=True)

    # --- RADAR CHART ---
    st.markdown("### 🕸️ DALY Distribution Radar")
    radar_df = df[["Age Group", "DALY Male", "DALY Female"]].set_index("Age Group")
    fig_radar = go.Figure()
    for col in radar_df.columns:
        fig_radar.add_trace(go.Scatterpolar(r=radar_df[col], theta=radar_df.index, fill='toself', name=col))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    st.plotly_chart(fig_radar, use_container_width=True)

# --- TAB 2: ML INSIGHTS ---
with tab2:
    st.subheader("📌 Anomaly Detection")
    def anomaly(column, label):
        score = zscore(df[column])
        idx = np.argmax(np.abs(score))
        return f"🔍 **{label} anomaly:** {df.loc[idx, 'Age Group']} — {df.loc[idx, column]:,.0f} DALY"

    if show_male: st.markdown(anomaly("DALY Male", "Male"))
    if show_female: st.markdown(anomaly("DALY Female", "Female"))
    if show_prediction and "DALY Male 2025" in df:
        st.markdown(f"📊 Predicted DALY peak in **{df.loc[df['DALY Male 2025'].idxmax(), 'Age Group']}**")

    if show_cluster:
        st.subheader("🧬 Clustering")
        cluster_data = df[["DALY Male", "DALY Female"]].dropna()
        kmeans = KMeans(n_clusters=3, n_init="auto").fit(cluster_data)
        df["Cluster"] = kmeans.labels_
        cluster_fig = px.scatter(df, x="DALY Male", y="DALY Female", color="Cluster", title="Gender DALY Clustering")
        st.plotly_chart(cluster_fig, use_container_width=True)

# --- TAB 3: RAW DATA & TOOLS ---
with tab3:
    st.subheader("📤 Upload Custom CSV")
    uploaded = st.file_uploader("Upload your DALY CSV", type="csv")
    if uploaded:
        user_df = pd.read_csv(uploaded)
        st.success("✅ File uploaded successfully!")
        st.dataframe(user_df.head())

    st.subheader("📥 Export")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("⬇️ Download CSV", df.to_csv(index=False), file_name="neuropulse_data.csv")
    with col2:
        img = BytesIO()
        fig.write_image(img, format="png")
        st.download_button("🖼️ Download Chart (PNG)", img.getvalue(), "neuropulse_chart.png", "image/png")

    st.subheader("🔎 View Full Data Table")
    st.dataframe(df.style.format(precision=0))

# --- BONUS: NATURAL LANGUAGE Q&A ---
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI

with st.expander("🗣️ Ask NeuroPulse AI"):
    query = st.text_input("Ask anything about the data...")
    if query:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        agent = create_pandas_dataframe_agent(ChatOpenAI(model="gpt-3.5-turbo"), df, verbose=False)
        with st.spinner("🤖 Thinking..."):
            st.success(agent.run(query))
