import streamlit as st
import pandas as pd
from model_utils import train_model, predict_next_year

st.set_page_config(page_title="NHIS Unmet Care Forecast", layout="centered")
st.title("NHIS: Forecast Delayed Care (Unmet Need)")

# ---- Load data ----
@st.cache_data
def load_data():
    return pd.read_csv("./preprocessed_data.csv")

df_clean = load_data()

# ---- UI inputs ----
topics = [
    "Delayed getting medical care due to cost among adults",
    "Did not get needed medical care due to cost",
    "Did not get needed mental health care due to cost"
]
topic = st.selectbox("Select topic", topics)
#st.caption(
#    "These indicators measure the percentage of adults who experienced cost-related barriers "
#    "to receiving medical or mental health care."
#)
category = st.selectbox("Select subgroup category", ["Age", "Insurance", "Income (FPL)"])

if category == "Age":
    subgroup = st.selectbox(
        "Select subgroup", 
        ["18-34 years", "50-64 years", "65 years and older"]
        )
elif category == "Insurance":
    subgroup = st.selectbox(
        "Select subgroup", 
        ["Uninsured", "Medicaid or other state programs", "Private"]
        )
else:
    subgroup = st.selectbox(
        "Select subgroup", 
        ["<100% FPL", "100% to <200% FPL", "≥200% FPL"]
        )

next_year = 2025
st.write("Prediction year:", next_year)

# ---- Predict ----
if st.button("Predict"):
    # Filtering based on TOPIC & SUBGROUP
    df_used = df_clean[(df_clean["TOPIC"] == topic) & (df_clean["SUBGROUP"] == subgroup)].sort_values("TIME_PERIOD")

    if len(df_used) < 3:
        st.error("Not enough data points for this topic/subgroup.")
    else:
        X = df_used[["TIME_PERIOD"]]
        y = df_used["ESTIMATE"]
        
        model = train_model(X, y, subgroup)

        first_year = X['TIME_PERIOD'].min()
        t_next = next_year - first_year

        pred = predict_next_year(model, t_next,subgroup)

        st.write(f"**Topic:** {topic}")
        st.write(f"**Subgroup:** {subgroup}")
        st.metric(f"Health Access Estimate in {next_year}", f"{pred:.2f}%")
        st.caption("This represents the estimated percentage of delayed care for the selected population, reflecting unmet healthcare needs.")


        chart_df = df_used[['TIME_PERIOD', 'ESTIMATE']].sort_values('TIME_PERIOD')
        chart_df = pd.concat(
            [chart_df, pd.DataFrame({'TIME_PERIOD':[next_year], 'ESTIMATE':[pred]})],
            ignore_index=True
        ).sort_values('TIME_PERIOD')

        chart_df = chart_df.set_index('TIME_PERIOD')

        st.subheader("Historical trend + next-year estimate")
        st.line_chart(chart_df)
        st.caption("Last point is the next-year estimate.")