import streamlit as st
from streamlit_option_menu import option_menu

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="HDIE",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)


# ======================================
# LOAD CSS
# ======================================

with open("assets/styles.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# ======================================
# RESTORE SIDEBAR BUTTON
# ======================================

st.markdown("""
<style>

[data-testid="collapsedControl"] {
    display: block !important;
    color: white !important;
    background-color: transparent !important;
}

</style>
""", unsafe_allow_html=True)

# ======================================
# SIDEBAR
# ======================================

with st.sidebar:

    st.markdown("""
    <h1 style='
        color:white;
        font-size:42px;
        margin-bottom:0px;
    '>
    HDIE
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style='
        color:#9CA3AF;
        margin-top:0px;
        margin-bottom:25px;
        font-size:15px;
    '>
    Human Decision Intelligence Engine
    </p>
    """, unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,

        options=[
            "Dashboard",
            "Predictions",
            "Segmentation",
            "Insights",
            "Explainability",
            "Performance"
        ],

        icons=[
            "speedometer2",
            "activity",
            "people",
            "lightbulb",
            "bar-chart",
            "graph-up"
        ],

        default_index=0,

        styles={

            "container": {
                "padding": "0!important",
                "background-color": "#0B1120",
            },

            "icon": {
                "color": "#60A5FA",
                "font-size": "18px"
            },

            "nav-link": {

                "font-size": "17px",

                "text-align": "left",

                "margin": "6px",

                "border-radius": "12px",

                "padding": "12px",

                "color": "white",

            },

            "nav-link-selected": {

                "background":
                "linear-gradient(90deg,#7C3AED,#2563EB)",

            },
        }
    )



# ======================================
# PAGE ROUTING
# ======================================
if selected == "Dashboard":

    st.markdown("""
    <h1 style="
    font-size:42px;
    font-weight:700;
    color:white;
    margin-bottom:5px;
    ">
    Human Decision Intelligence Dashboard
    </h1>

    <p style="
    color:#94A3B8;
    font-size:16px;
    ">
    AI-powered behavioral analytics and explainable intelligence platform
    </p>
    """, unsafe_allow_html=True)

    exec(open("app/dashboard.py", encoding="utf-8").read())


elif selected == "Predictions":
    exec(
    open(
        "app/predictions.py",
        encoding="utf-8"
    ).read()
)

elif selected == "Segmentation":
    exec(open("app/segmentation.py", encoding="utf-8").read())

elif selected == "Insights":
    exec(open("app/business_recommendations.py", encoding="utf-8").read())

elif selected == "Explainability":
    exec(open("app/explainability.py", encoding="utf-8").read())

elif selected == "Performance":

    exec(open(
        "app/performance.py",
        encoding="utf-8"
    ).read())
