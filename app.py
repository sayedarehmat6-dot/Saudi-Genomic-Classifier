import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Saudi Genomic Classifier (Chr21)",
    page_icon="🧬",
    layout="wide"
)

# -----------------------------
# CUSTOM STYLING (PROFESSIONAL LOOK)
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
    color: #fafafa;
}
h1, h2, h3 {
    color: #00d4ff;
}
.stButton>button {
    background-color: #00d4ff;
    color: black;
    border-radius: 8px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

FEATURES = ["pos", "global_af", "impact", "pli", "loeuf"]

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    path = os.path.join("models", "pathogenicity_model.json")
    if not os.path.exists(path):
        st.error("Model file missing (models/pathogenicity_model.json)")
        return None
    model = xgb.XGBClassifier()
    model.load_model(path)
    return model

model = load_model()

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    path = os.path.join("data", "Saudi_Variant_AI_Master_Dataset.csv")
    if not os.path.exists(path):
        st.error("Dataset missing (data/Saudi_Variant_AI_Master_Dataset.csv)")
        return None
    return pd.read_csv(path)

df = load_data()

# Detect label column
label_col = None
if df is not None:
    for col in df.columns:
        if col.lower() in ["label", "class", "target", "y"]:
            label_col = col
            break

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.title(" SGC v1.0")

    st.markdown("""
**Population-Aware Variant Predictor**

**Data Sources:** gnomAD • ClinVar • Saudi PAVS 
""")
**Scope:** Chr21 Pilot *(not genome-wide)*

**Model:** XGBoost (~200 trees)

**Features:**
- Allele Frequency  
- Functional Impact  
- pLI (constraint)  
- LOEUF (constraint)
    """)

    st.divider()

    st.markdown("""
**Hypothesis:**  
Rare variants in highly constrained genes  
are more likely pathogenic.
    """)

    st.divider()

    st.caption("Research Prototype • Not for clinical use")

# -----------------------------
# HEADER
# -----------------------------
st.title(" Saudi Genomic Classifier (Chr21 Pilot)")
st.markdown("AI-driven pathogenicity prediction for Chromosome 21 variants")
st.markdown("---")

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "Variant Analysis",
    "Dataset Explorer",
    "Model Insights"
])

# =====================================================
# 🔬 VARIANT ANALYSIS
# =====================================================
with tab1:

    st.subheader("Variant Input")

    col1, col2, col3 = st.columns(3)

    with col1:
        pos = st.number_input("Genomic Position", value=10000000)
        impact = st.selectbox("Impact", ["HIGH", "MODERATE", "LOW", "MODIFIER"])

    with col2:
        global_af = st.number_input(
            "Global AF",
            min_value=0.0,
            max_value=1.0,
            value=0.0001,
            format="%.6f"
        )
        pli = st.slider("pLI", 0.0, 1.0, 0.5)

    with col3:
        loeuf = st.slider("LOEUF", 0.0, 2.0, 0.5)

    impact_map = {'HIGH': 3, 'MODERATE': 2, 'LOW': 1, 'MODIFIER': 0}
    impact_val = impact_map[impact]

    if st.button("Run Analysis"):

        if model is None:
            st.error("Model not loaded.")
        else:
            input_df = pd.DataFrame([{
                "pos": pos,
                "global_af": global_af,
                "impact": impact_val,
                "pli": pli,
                "loeuf": loeuf
            }])[FEATURES]

            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0]

            c1, c2 = st.columns([1,2])

            with c1:
                if pred == 1:
                    st.error("### PATHOGENIC")
                    st.metric("Confidence", f"{prob[1]*100:.2f}%")
                else:
                    st.success("### BENIGN / VUS")
                    st.metric("Confidence", f"{prob[0]*100:.2f}%")

                st.subheader("Biological Interpretation")

                if global_af < 0.001:
                    st.write("• Rare variant → supports pathogenicity")

                if pli > 0.9:
                    st.write("• High gene constraint (pLI)")

                if loeuf < 0.5:
                    st.write("• Strong intolerance (low LOEUF)")

            with c2:
                fig = go.Figure(go.Bar(
                    x=[prob[0], prob[1]],
                    y=["Benign", "Pathogenic"],
                    orientation='h'
                ))
                fig.update_layout(title="Prediction Confidence")
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Input Features")
            st.dataframe(input_df)

# =====================================================
# 📊 DATASET EXPLORER
# =====================================================
with tab2:

    st.info("Dataset contains Chromosome 21 variants only.")

    if df is not None:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(100))

        if label_col:
            st.subheader("Class Distribution")
            st.bar_chart(df[label_col].value_counts())

        if all(c in df.columns for c in ["global_af", "pli"]):
            st.subheader("AF vs Constraint")

            fig = px.scatter(
                df.sample(min(1000, len(df))),
                x="global_af",
                y="pli",
                color=label_col if label_col else None
            )
            st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 🧠 MODEL INSIGHTS
# =====================================================
with tab3:

    st.subheader("Model Overview")

    st.write("""
- XGBoost classifier  
- ~200 trees  
- Binary classification  
- Chr21-specific training dataset  
    """)

    if model is not None:
        st.subheader("Feature Importance")

        booster = model.get_booster()
        imp = booster.get_score(importance_type='weight')

        if imp:
            imp_df = pd.DataFrame({
                "Feature": list(imp.keys()),
                "Importance": list(imp.values())
            }).sort_values(by="Importance", ascending=True)

            fig = px.bar(
                imp_df,
                x="Importance",
                y="Feature",
                orientation='h'
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("SHAP Interpretability")

    if os.path.exists("images/shap_summary_bar.png"):
        st.image("images/shap_summary_bar.png")

    if os.path.exists("images/shap_beeswarm.png"):
        st.image("images/shap_beeswarm.png")

    st.subheader("Scope & Limitation")

    st.write("""
This model is trained only on Chromosome 21 variants  
and serves as a pilot for population-aware prediction.

Not intended for genome-wide or clinical use.
    """)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:gray;'>"
    "Saudi Genomic Classifier (Chr21 Pilot) | Sayeda Rehmat"
    "</div>",
    unsafe_allow_html=True
)
