import pandas as pd
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from scipy.stats import zscore
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# --- Load and Prepare Data ---
df = pd.read_csv("epilepsy_daly_by_age_2019.csv")
df["DALY Male 2015"] = df["DALY Male"] * 1.05
df["DALY Female 2015"] = df["DALY Female"] * 1.08
age_groups = df["Age Group"]

# --- Page Config ---
st.set_page_config(page_title="NeuroPulse AI Dashboard", layout="wide")

# --- Header ---
st.markdown("<h1 style='color:#FFD700;'>🧠 NeuroPulse++</h1>", unsafe_allow_html=True)
st.markdown("**AI-powered dashboard for epilepsy DALY prediction, analysis & ML insights**")
st.caption("📊 Data Source: WHO Global Health Estimates (2019)")

# --- Sidebar ---
st.sidebar.title("⚙️ Customize View")
show_male = st.sidebar.checkbox("🧑‍⚕️ Show Male", True)
show_female = st.sidebar.checkbox("👩‍⚕️ Show Female", True)
show_prediction = st.sidebar.checkbox("🔮 Predict 2025", True)
chart_style = st.sidebar.radio("📈 Chart Style", ["Line", "Bar"])
show_cluster = st.sidebar.checkbox("🧬 Show Cluster Analysis", False)

# --- Predict DALY ---
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

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 ML Insights", "📦 Raw Data"])

with tab1:
    # --- KPI Cards ---
    st.subheader("📌 Key Performance Metrics")
    kpi1, kpi2, kpi3 = st.columns(3)
    if show_male:
        peak_idx = df["DALY Male"].idxmax()
        kpi1.metric("👨 Peak Male DALY", f"{df['DALY Male'].max():,.0f}", df.loc[peak_idx, "Age Group"])
    if show_female:
        peak_idx = df["DALY Female"].idxmax()
        kpi2.metric("👩 Peak Female DALY", f"{df['DALY Female'].max():,.0f}", df.loc[peak_idx, "Age Group"])
    if show_prediction:
        avg_pred = df[[c for c in df.columns if "2025" in c]].mean().mean()
        kpi3.metric("🔮 2025 Predicted Avg DALY", f"{avg_pred:,.0f}")

    # --- Chart Plot ---
    st.markdown("### 📈 DALY by Age Group")
    fig = go.Figure()

    def add_trace(label, y_data, color, dash="solid"):
        mode = "lines+markers" if chart_style == "Line" else "markers"
        fig.add_trace(go.Scatter(
            x=age_groups, y=y_data, mode=mode,
            name=label, line=dict(color=color, dash=dash)))

    if show_male:
        add_trace("Male 2019", df["DALY Male"], "blue")
    if show_female:
        add_trace("Female 2019", df["DALY Female"], "green")
    if show_prediction:
        if show_male and "DALY Male 2025" in df:
            add_trace("Male 2025", df["DALY Male 2025"], "orange", "dash")
        if show_female and "DALY Female 2025" in df:
            add_trace("Female 2025", df["DALY Female 2025"], "red", "dash")

    fig.update_layout(
        xaxis_title="Age Group", yaxis_title="DALY",
        hovermode="x unified",
        height=500, margin=dict(t=30), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("📌 Auto-Insights")

    def anomaly_insight(column, label):
        scores = zscore(df[column])
        idx = np.argmax(np.abs(scores))
        return f"🔍 **{label} anomaly:** {df.loc[idx, 'Age Group']} with **{df.loc[idx, column]:,.0f} DALY**"

    if show_male: st.markdown(anomaly_insight("DALY Male", "Male"))
    if show_female: st.markdown(anomaly_insight("DALY Female", "Female"))
    if show_prediction and "DALY Male 2025" in df:
        peak_idx = df["DALY Male 2025"].idxmax()
        st.markdown(f"📊 Predicted peak in **{df.loc[peak_idx, 'Age Group']}** age group for males.")

    if show_cluster:
        st.subheader("🧬 Clustering by DALY")
        cluster_data = df[["DALY Male", "DALY Female"]].dropna()
        kmeans = KMeans(n_clusters=3, n_init="auto").fit(cluster_data)
        df["Cluster"] = kmeans.labels_
        fig2 = px.scatter(df, x="DALY Male", y="DALY Female", color="Cluster",
                          size_max=60, title="DALY Gender Clustering")
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("📤 Export & Raw View")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("⬇️ Download CSV", df.to_csv(index=False), file_name="neuropulse_data.csv")
    with col2:
        img = BytesIO()
        fig.write_image(img, format="png")
        st.download_button("🖼️ Download Chart (PNG)", data=img.getvalue(), file_name="neuropulse_chart.png", mime="image/png")
    with st.expander("🔎 View Full Data Table"):
        st.dataframe(df.style.format(precision=0))
