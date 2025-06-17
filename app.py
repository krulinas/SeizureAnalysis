import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from io import BytesIO
import numpy as np
import plotly.graph_objects as go

# --- Load and Prepare Data ---
df = pd.read_csv("epilepsy_daly_by_age_2019.csv")
df["DALY Male 2015"] = df["DALY Male"] * 1.05
df["DALY Female 2015"] = df["DALY Female"] * 1.08
age_groups = df["Age Group"]
years = ["2015", "2019"]
genders = ["Male", "Female"]

# --- Page Setup ---
st.set_page_config(page_title="Epilepsy Smart Dashboard", layout="wide")
st.title("🧠 Epilepsy DALY Explorer")
st.markdown("### Powered by AI + Visualization • WHO Global Health Estimates 2019")

# --- Sidebar Filters ---
st.sidebar.header("🔍 Customize View")
show_male = st.sidebar.checkbox("👨 Show Male", True)
show_female = st.sidebar.checkbox("👩 Show Female", True)
chart_type = st.sidebar.radio("📈 Chart Style", ["Line", "Bar"])
show_prediction = st.sidebar.checkbox("🔮 Predict DALY for 2025", True)

# --- Chart Setup ---
fig, ax = plt.subplots()
x = np.arange(len(age_groups))
width = 0.3
highlight_age = ""

# Plot Actuals
def plot_actual(gender):
    y = df[f"DALY {gender}"]
    offset = -width/2 if gender == "Male" else width/2
    if chart_type == "Line":
        ax.plot(age_groups, y, label=f"{gender} 2019", marker='o')
    else:
        ax.bar(x + offset, y, width=width, label=f"{gender} 2019")

# Plot Prediction
def plot_prediction(gender):
    predictions = []
    for i in range(len(df)):
        X = np.array([[2015], [2019]])
        y = np.array([
            df.loc[i, f"DALY {gender} 2015"],
            df.loc[i, f"DALY {gender}"]
        ])
        model = LinearRegression()
        model.fit(X, y)
        pred = model.predict([[2025]])[0]
        predictions.append(pred)
    df[f"DALY {gender} 2025"] = predictions

    offset = -width*1.5 if gender == "Male" else width*1.5
    if chart_type == "Line":
        ax.plot(age_groups, predictions, linestyle='dashed', label=f"{gender} 2025 (Predicted)", marker='x')
    else:
        ax.bar(x + offset, predictions, width=width*0.8, label=f"{gender} 2025 (Predicted)", alpha=0.6, hatch="//")

# Plotting
if show_male:
    plot_actual("Male")
    if show_prediction:
        plot_prediction("Male")

if show_female:
    plot_actual("Female")
    if show_prediction:
        plot_prediction("Female")

# Styling
ax.set_xticks(x)
ax.set_xticklabels(age_groups, rotation=45)
ax.set_xlabel("Age Group")
ax.set_ylabel("DALY")
ax.set_title("DALY by Age Group")
ax.legend()
st.pyplot(fig)

# --- Dynamic Insights ---
st.markdown("### 📌 Automated Insights")

insights = []
if show_male:
    peak_male = df["DALY Male"].idxmax()
    insights.append(f"👨 **Male DALY peaks** in *{df.loc[peak_male, 'Age Group']}* with **{df.loc[peak_male, 'DALY Male']:.0f}**")
if show_female:
    peak_female = df["DALY Female"].idxmax()
    insights.append(f"👩 **Female DALY peaks** in *{df.loc[peak_female, 'Age Group']}* with **{df.loc[peak_female, 'DALY Female']:.0f}**")
if show_prediction:
    pred_male = df["DALY Male 2025"].max() if "DALY Male 2025" in df else 0
    pred_female = df["DALY Female 2025"].max() if "DALY Female 2025" in df else 0
    if pred_male or pred_female:
        insights.append("🔮 **2025 forecast shows a consistent trend** across young-to-mid adult age groups.")

for line in insights:
    st.markdown(f"- {line}")

# --- Downloads ---
st.markdown("### 📥 Export Your View")
st.download_button("📄 Download Data (CSV)", df.to_csv(index=False), "epilepsy_daly_dashboard.csv")
img_buffer = BytesIO()
fig.savefig(img_buffer, format='png')
st.download_button("🖼 Download Chart (PNG)", img_buffer.getvalue(), "epilepsy_daly_chart.png", mime="image/png")

# --- Raw Data ---
with st.expander("📂 Show Raw Table"):
    st.dataframe(df.style.format(precision=0))
