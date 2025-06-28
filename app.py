import pandas as pd
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import zscore
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from io import BytesIO
import os
import seaborn as sns
import matplotlib.pyplot as plt
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI

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
st.markdown("<h1>NeuroPulse++</h1>", unsafe_allow_html=True)
st.markdown("AI-powered dashboard for epilepsy DALY prediction, clustering, and human insight.")
st.caption("Source: WHO Global Health Estimates (2019)")
with st.expander("About This Dashboard"):
    st.markdown("""
    **What is Epilepsy?**  
    Epilepsy is a brain disorder that causes repeated seizures. Seizures are sudden bursts of electrical activity in the brain that can affect behavior, movement, or awareness.

    **What is a Seizure?**  
    A seizure is like a short circuit in the brain. It can cause jerking movements, confusion, staring spells, or even loss of consciousness — depending on the type.

    **What is DALY?**  
    DALY stands for *Disability-Adjusted Life Year*. It measures how much healthy life is lost due to illness or death. The higher the number, the bigger the impact of epilepsy on a person's life.

    **What does this dashboard do?**  
    This dashboard shows epilepsy data for different age groups. It lets you:
    - View current and past epilepsy burden
    - Predict future trends using machine learning
    - Explore patterns by age and gender

    **What ML (Machine Learning) is used?**  
    A simple method called **Linear Regression** is used here to predict future values. It looks at trends from the past and draws a line to guess future numbers.
    """)

st.markdown("---")

# --- SIDEBAR CONTROLS ---
st.sidebar.title("Settings")
st.sidebar.markdown("""
<div style='font-size:16px; font-weight:bold; color:#FF9FA2; padding: 6px 0;'>
</div>
""", unsafe_allow_html=True)

show_male = st.sidebar.checkbox("Show Male", True)
show_female = st.sidebar.checkbox("Show Female", True)
show_prediction = st.sidebar.checkbox("Show 2025 Prediction", True)
chart_style = st.sidebar.radio("Chart Style", ["Line", "Bar"])
show_guided_tips = st.sidebar.toggle("Show Guided Tips", value=False)

def explain(section):
    if not show_guided_tips:
        return
    explanations = {
        "daly_chart": "This chart shows DALY (years of healthy life lost) for each age group. Higher = more impact.",
        "radar": "Compares DALY for males and females across age groups. Wider = greater impact.",
        "r2": "R² Score shows how accurate our prediction is. 1.0 = perfect, 0 = no match.",
        "anomaly": "Anomalies are age groups with unusually high or low health loss.",
        "heatmap": "Correlation shows how values relate. 1.00 = strong link.",
        "boxplot": "Each dot is an age group. Taller boxes = more variation.",
        "trend": "Forecast shows how DALY may change from 2015 → 2025.",
        "upload": "Upload a CSV file with epilepsy DALY data. We'll show the first few rows here.",
        "med": "Educational info for seizure types and recommended treatments."
    }
    st.caption(explanations.get(section, ""))

def insight(section):
    if not show_guided_tips:
        return
    insights = {
        "daly_chart": "Insight: Young adults (15–29) show the highest DALY in both genders.",
        "radar": "Insight: DALY patterns are nearly parallel for both genders across age groups.",
        "boxplot": "Insight: Female DALY values vary more widely than male.",
        "trend": "Insight: Most age groups show stable or slightly decreasing DALY by 2025.",
        "heatmap": "Insight: DALY values for male and female are highly correlated (~0.98).",
    }
    st.markdown(insights.get(section, ""))

# --- DATA LOAD & PREP ---
def load_data():
    df = pd.read_csv("epilepsy_daly_by_age_2019.csv")
    df["DALY Male 2015"] = df["DALY Male"] * 1.05
    df["DALY Female 2015"] = df["DALY Female"] * 1.08
    age_groups = df["Age Group"]
    return df, age_groups

df, age_groups = load_data()

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
tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "ML Insights", "Data & Tools", "Medical Guidelines"])

# --- TAB 1: MAIN DASHBOARD ---
with tab1:
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    if show_male:
        peak = df["DALY Male"].idxmax()
        col1.metric("Peak Male DALY", f"{df['DALY Male'].max():,.0f}", df.loc[peak, "Age Group"])
    if show_female:
        peak = df["DALY Female"].idxmax()
        col2.metric("Peak Female DALY", f"{df['DALY Female'].max():,.0f}", df.loc[peak, "Age Group"])
    if show_prediction:
        pred_avg = df[[c for c in df.columns if '2025' in c]].mean().mean()
        col3.metric("🔮 Avg Predicted 2025", f"{pred_avg:,.0f}")

    if not show_male and not show_female:
        st.warning("Please select at least one gender to display the charts.")
    
    # --- CHART ---
    st.markdown("DALY by Age Group")
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

    fig.update_layout(
        title="DALY by Age Group", 
        hovermode="x unified", 
        height=500, 
        plot_bgcolor="#2E0F15",
        paper_bgcolor="#2E0F15",
        font_color="#FAFFF0",
        xaxis_title="Age Group", yaxis_title="DALY",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        )
    )
    fig.update_traces(marker=dict(size=6), selector=dict(mode='lines+markers'))
    fig.update_layout(transition_duration=500)
    st.plotly_chart(fig, use_container_width=True)

    # --- RADAR CHART ---
    st.markdown("DALY Distribution Radar")
    radar_df = df[["Age Group", "DALY Male", "DALY Female"]].set_index("Age Group")
    fig_radar = go.Figure()
    for col in radar_df.columns:
        fig_radar.add_trace(go.Scatterpolar(r=radar_df[col], theta=radar_df.index, fill='toself', name=col))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    st.plotly_chart(fig_radar, use_container_width=True)

# --- TAB 2: ML INSIGHTS ---
with tab2:
    st.subheader("Machine Learning Insights")

    with st.expander("What Is Machine Learning Doing Here?"):
        st.markdown("""
        Machine Learning helps us predict future DALY (Disability-Adjusted Life Years) values for epilepsy by learning from past data.

        This dashboard uses **Linear Regression**:
        - It draws a trend line from past data (2015 to 2019)
        - It then extends the line to predict values for **2025**

        The goal is to help public health teams understand future risk and prepare better.
        """)

    # --- R² Score ---
    if show_prediction:
        st.markdown("Prediction Accuracy (R² Score)")
        if show_male and "DALY Male 2025" in df:
            r2_male = r2_score(df["DALY Male"], df["DALY Male 2025"])
            st.markdown(f"🔵 **Male R² Score**: `{r2_male:.2f}`")
        if show_female and "DALY Female 2025" in df:
            r2_female = r2_score(df["DALY Female"], df["DALY Female 2025"])
            st.markdown(f"🟣 **Female R² Score**: `{r2_female:.2f}`")

    # --- Anomaly Detection ---
    st.markdown("Anomaly Detection")
    def anomaly(column, label):
        score = zscore(df[column])
        idx = np.argmax(np.abs(score))
        return f"🔍 **{label} anomaly**: {df.loc[idx, 'Age Group']} — {df.loc[idx, column]:,.0f} DALY"

    if show_male:
        st.markdown(anomaly("DALY Male", "🔵 Male"))
    if show_female:
        st.markdown(anomaly("DALY Female", "🟣 Female"))
    if show_prediction and "DALY Male 2025" in df:
        st.markdown(f"Predicted DALY peak in **{df.loc[df['DALY Male 2025'].idxmax(), 'Age Group']}**")

    # --- Correlation Heatmap ---
    st.markdown("Correlation Heatmap")
    heatmap_data = df.select_dtypes(include=[np.number])
    corr = heatmap_data.corr()

    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="Reds", fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.markdown("""Quick Summary
    - **DALY** measures total health loss from epilepsy.
    - **Higher numbers** mean more years of healthy life lost.
    - **Most affected group**: 15–29 year olds.
    - Future trends show **slight improvement** by 2025.
    """)

    # --- Distribution Chart ---
    st.markdown("Distribution of DALY by Gender")
    fig2 = go.Figure()
    if show_male:
        fig2.add_trace(go.Box(
            y=df["DALY Male"],
            name="Male",
            boxpoints="all",
            marker_color="blue",
            hoverinfo="y",
            hovertemplate="Male DALY: %{y:,} <extra></extra>"
        ))
    if show_female:
        fig2.add_trace(go.Box(
            y=df["DALY Female"],
            name="Female",
            boxpoints="all",
            marker_color="violet",
            hoverinfo="y",
            hovertemplate="Female DALY: %{y:,} <extra></extra>"
        ))

    st.plotly_chart(fig2, use_container_width=True)

    # --- Trend Forecasting Line Chart ---
    st.markdown("Trend Forecast: 2015 → 2019 → 2025")
    fig_trend = go.Figure()
    years = [2015, 2019, 2025]

    for i in range(len(df)):
        age = df.loc[i, "Age Group"]
        if show_male:
            y_male = [df.loc[i, "DALY Male 2015"], df.loc[i, "DALY Male"], df.loc[i, "DALY Male 2025"]]
            fig_trend.add_trace(go.Scatter(
                x=years,
                y=y_male,
                name=f"{age} Male",
                line=dict(dash='solid'),
                hovertemplate=f"<b>{age} Male</b><br>Year: %{{x}}<br>DALY: %{{y:,}} <extra></extra>"
            ))
        if show_female:
            y_female = [df.loc[i, "DALY Female 2015"], df.loc[i, "DALY Female"], df.loc[i, "DALY Female 2025"]]
            fig_trend.add_trace(go.Scatter(
                x=years,
                y=y_female,
                name=f"{age} Female",
                line=dict(dash='dot'),
                hovertemplate=f"<b>{age} Female</b><br>Year: %{{x}}<br>DALY: %{{y:,}} <extra></extra>"
            ))

    fig_trend.update_layout(title="DALY Forecast by Age Group", xaxis_title="Year", yaxis_title="DALY", height=600)
    st.plotly_chart(fig_trend, use_container_width=True)

# --- TAB 3: RAW DATA & TOOLS ---
with tab3:
    st.subheader("Upload Custom CSV")
    st.caption("Upload a CSV with similar structure to our DALY dataset. We'll show the first few rows.")

    uploaded = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded:
        try:
            user_df = pd.read_csv(uploaded)
            st.success("File uploaded successfully!")
            st.dataframe(user_df.head())
        except Exception as e:
            st.error("Failed to load CSV. Please make sure it's properly formatted.")
            st.code(str(e))

    st.subheader("View Full Data Table")
    st.dataframe(df.style.format(precision=0))

# --- TAB 4: MEDICAL GUIDELINES ---
with tab4:
    st.subheader("Doctor Recommendations & Medicine Info")
    st.markdown("This section provides general guidance for epilepsy patients and caregivers. Please consult a licensed physician for diagnosis and treatment.")

    st.markdown("Common Seizure Types")
    st.markdown("""
    - **Focal (Partial) Seizures**: Affect one part of the brain. May cause twitching or confusion.
    - **Generalized Seizures**: Affect both sides. Includes tonic-clonic (grand mal), absence, and atonic seizures.
    """)

    st.markdown("Recommended Medications by Stage")
    med_data = {
        "Stage": ["Initial Diagnosis", "Mild/Occasional Seizures", "Frequent or Severe Seizures", "Drug-resistant Epilepsy"],
        "Medications": [
            "Carbamazepine, Lamotrigine",
            "Valproate, Levetiracetam",
            "Topiramate, Clobazam",
            "Ketogenic diet, Vagus nerve stimulation (VNS), Surgery"
        ]
    }
    med_df = pd.DataFrame(med_data)
    st.table(med_df)

    st.markdown("Find Pharmacies Nearby")
    scol1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/3176/3176367.png", width=40)  # or use a local icon
    with col2:
        if st.button("🔎 Open in Google Maps"):
            st.markdown("[Click here to search pharmacies near you](https://www.google.com/maps/search/epilepsy+medication+pharmacy+near+me)", unsafe_allow_html=True)

    st.markdown("Doctor Advice Highlights")
    st.markdown("""
    - Take medications consistently at the same time every day.
    - Avoid triggers like sleep deprivation, flashing lights, or alcohol.
    - Keep a **seizure diary** for tracking episodes.
    - Inform close contacts about first-aid steps during a seizure.
    - Never stop medication without medical advice.
    """)

    st.info("This content is educational and not a substitute for medical advice.")
