import streamlit as st
import json
import os

# ======================================
# TITLE
# ======================================

st.markdown(
    "<h1 style='margin-top:0;'>Model Performance</h1>",
    unsafe_allow_html=True
)

st.caption(
    "Evaluation metrics and model validation results"
)

st.markdown("---")

# ======================================
# LOAD METRICS
# ======================================

metrics = {}

metrics_path = "reports/model_metrics.json"

if os.path.exists(metrics_path):

    with open(metrics_path) as f:

        metrics = json.load(f)

else:

    st.error(
        "model_metrics.json not found. Run training pipeline again."
    )

# ======================================
# METRIC CARDS
# ======================================

if metrics:

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Accuracy",
            f"{metrics['accuracy']*100:.2f}%"
        )

    with col2:
        st.metric(
            "Precision",
            f"{metrics['precision']*100:.2f}%"
        )

    with col3:
        st.metric(
            "Recall",
            f"{metrics['recall']*100:.2f}%"
        )

    with col4:
        st.metric(
            "F1 Score",
            f"{metrics['f1']*100:.2f}%"
        )

    with col5:
        st.metric(
            "ROC AUC",
            f"{metrics['roc_auc']:.3f}"
        )

st.markdown("---")

# ======================================
# VISUAL EVALUATION
# ======================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("Confusion Matrix")

    cm_path = "reports/confusion_matrix.png"

    if os.path.exists(cm_path):

        st.image(
            cm_path,
            use_container_width=True
        )

    else:

        st.warning(
            "Confusion Matrix image not found"
        )

with col2:

    st.subheader("ROC Curve")

    roc_path = "reports/roc_curve.png"

    if os.path.exists(roc_path):

        st.image(
            roc_path,
            use_container_width=True
        )

    else:

        st.warning(
            "ROC Curve image not found"
        )

st.markdown("---")

# ======================================
# PERFORMANCE INTERPRETATION
# ======================================

st.subheader(
    "Performance Interpretation"
)

if metrics:

    st.info(
        f"""
The purchase prediction model achieved an overall accuracy of
{metrics['accuracy']*100:.2f}% on unseen test data.

The ROC-AUC score of {metrics['roc_auc']:.3f}
indicates the model's ability to distinguish between
purchasing and non-purchasing users.

Precision of {metrics['precision']*100:.2f}% shows
how reliable positive purchase predictions are.

Recall of {metrics['recall']*100:.2f}% reflects
the model's ability to identify actual purchasing users.

The F1 Score of {metrics['f1']*100:.2f}% demonstrates
the balance between precision and recall.
"""
    )

# ======================================
# MODEL ASSESSMENT
# ======================================

st.subheader(
    "Model Assessment"
)

if metrics:

    if metrics["roc_auc"] >= 0.75:

        st.success(
            """
The model demonstrates strong predictive performance
and is suitable for behavioral analytics and
decision-support applications.
"""
        )

    elif metrics["roc_auc"] >= 0.65:

        st.warning(
            """
The model demonstrates moderate predictive capability.
Further feature engineering may improve performance.
"""
        )

    else:

        st.error(
            """
The model requires improvement before deployment.
"""
        )