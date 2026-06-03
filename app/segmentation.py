import streamlit as st
import pandas as pd
import plotly.express as px

st.markdown(
    "<h1 style='margin-top:0;'>User Segments</h1>",
    unsafe_allow_html=True
)




clustered_df = pd.read_csv(
    "data/processed/clustered_data.csv"
)

segment_counts = (
    clustered_df['user_segment']
    .value_counts()
)

col1,col2 = st.columns([2,1])

with col1:

    fig = px.bar(
        x=segment_counts.index,
        y=segment_counts.values,
        title="User Segment Distribution",
        color=segment_counts.values,
        color_continuous_scale="Blues"
    )

    fig.update_layout(
        height=450,
        template="plotly_dark"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    st.metric(
        "Segments",
        len(segment_counts)
    )

    for seg,count in segment_counts.items():

        st.metric(seg,count)



st.dataframe(
    clustered_df.head(20),
    use_container_width=True
)