import streamlit as st
import pandas as pd

st.title("Behavioral Bias Insights")

st.markdown("---")

bias_df = pd.read_csv(
    "data/processed/bias_analysis.csv"
)

bias_columns = [
    'loss_aversion',
    'comparison_overload',
    'decision_fatigue',
    'impulsive_behavior'
]

bias_counts = bias_df[bias_columns].sum()

st.subheader("Detected Behavioral Biases")

for bias, count in bias_counts.items():

    st.metric(
        bias.replace("_", " ").title(),
        int(count)
    )

st.markdown("---")

st.info("""
Behavioral bias analysis helps businesses understand:
- why users abandon carts
- why users hesitate
- how decision fatigue affects purchases
- how impulsive behavior impacts conversion
""")