import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Saudi Genomic Classifier (Chr21 Pilot)",
    page_icon="🧬",
    layout="wide"
)

FEATURES = ["pos", "global_af", "impact", "pli", "loeuf"]


# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join("models", "pathogenicity_model.json")
    if not os.path.exists(model_path):
        st.error(f"Model not found at: {model_path}")
        return None
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model

model = load_model()

# -----------------------------
# LOAD DATASET
# -----------------------------
@st.cache_data
def load_data():
    data_path = os.path.join("data", "Saudi_Variant_AI_Master_Dataset.csv")
    if not os.path.exists(data_path):
        st.error(f"Dataset not found at: {data_path}")
        return None
    return pd.read_csv(data_path)

df = load_data()

# Auto-detect label column
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
    st.title("🧬 SGC v1.0")

    st.markdown("""
**Developer:** Sayeda Rehmat  

**Project:**  
Population-Aware Variant Pathogenicity Prediction (Saudi Focus)

**Model:** XGBoost (~200 Trees)

**Core Features:**
- Allele Frequency
- Gene Constraint (pLI, LOEUF)
- Functional Impact
    """)
st.markdown("""
**Scope:**  
Chromosome 21 (Pilot Study)

This model is trained only on Chr21 variants and is not genome-wide.
""")
    st.divider()

    st.markdown("""
**Hypothesis:**  
Rare variants in constrained genes are more likely pathogenic.
    """)

    st.divider()

    st.markdown("""
 
- No clinical validation  
- Research prototype only  
    """)

# -----------------------------
# MAIN TITLE
# -----------------------------
st.title("🧬 Saudi Genomic Classifier (Chr21 Pilot)")
st.markdown("---")

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "🔬 Variant Analysis",
    "📊 Dataset Explorer",
    "🧠 Model Insights"
])

# =========================================================
# 🔬 VARIANT ANALYSIS
# =========================================================
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
            format="%.6f",
            value=0.0001
        )
        pli = st.slider("pLI", 0.0, 1.0, 0.5)

    with col3:
        loeuf = st.slider("LOEUF", 0.0, 2.0, 0.5)

    impact_map = {'HIGH': 3, 'MODERATE': 2, 'LOW': 1, 'MODIFIER': 0}
    impact_val = impact_map[impact]

    if st.button("Run Analysis", use_container_width=True):

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

            colA, colB = st.columns([1, 2])

            with colA:
                if pred == 1:
                    st.error("### PATHOGENIC")
                    st.metric("Confidence", f"{prob[1]*100:.2f}%")
                else:
                    st.success("### BENIGN / VUS")
                    st.metric("Confidence", f"{prob[0]*100:.2f}%")

                st.subheader("Interpretation")

                if global_af < 0.001:
                    st.write("• Rare variant → higher pathogenic likelihood")

                if pli > 0.9:
                    st.write("• High gene constraint (pLI)")

                if loeuf < 0.5:
                    st.write("• Low LOEUF → strong intolerance")

            with colB:
                fig = go.Figure(go.Bar(
                    x=[prob[0], prob[1]],
                    y=['Benign', 'Pathogenic'],
                    orientation='h'
                ))
                fig.update_layout(title="Prediction Confidence")
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Feature Input Used")
            st.dataframe(input_df)

# =========================================================
# 📊 DATASET EXPLORER
# =========================================================
st.info("Dataset contains variants from Chromosome 21 only.")
with tab2:

    if df is None:
        st.warning("Dataset not loaded.")
    else:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(100))

        if label_col:
            st.subheader("Class Distribution")
            st.bar_chart(df[label_col].value_counts())

        if all(col in df.columns for col in ["global_af", "pli"]):
            st.subheader("Feature Relationship")

            fig = px.scatter(
                df.sample(min(1000, len(df))),
                x="global_af",
                y="pli",
                color=label_col if label_col else None,
                title="Allele Frequency vs Gene Constraint"
            )
            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 🧠 MODEL INSIGHTS
# =========================================================
with tab3:

    st.subheader("Model Overview")

    st.write("""
- Algorithm: XGBoost Classifier  
- Trees: ~200  
- Features: 5  
- Task: Binary Classification  
    """)
    st.subheader("Scope of Model")

st.write("""
This model was trained exclusively on Chromosome 21 variants as a pilot study. 
It is designed to demonstrate population-aware pathogenicity prediction and is not intended for genome-wide inference.
""")

    if model is not None:
        st.subheader("Feature Importance")

        booster = model.get_booster()
        importance = booster.get_score(importance_type='weight')

        if importance:
            imp_df = pd.DataFrame({
                "Feature": list(importance.keys()),
                "Importance": list(importance.values())
            }).sort_values(by="Importance", ascending=True)

            fig = px.bar(
                imp_df,
                x="Importance",
                y="Feature",
                orientation='h',
                title="Model Feature Importance"
            )

            st.plotly_chart(fig, use_container_width=True)

    st.subheader("SHAP Interpretability")

    if os.path.exists("images/shap_summary_bar.png"):
        st.image("images/shap_summary_bar.png")

    if os.path.exists("images/shap_beeswarm.png"):
        st.image("images/shap_beeswarm.png")

    st.subheader("Training Insight")

    st.write("""
The model uses biologically meaningful features:

- Allele frequency → rarity signal  
- pLI → gene intolerance  
- LOEUF → functional constraint  

These reflect evolutionary pressure and help distinguish pathogenic variants.
    """)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:gray;'>"
    "Saudi Genomic Classifier (Chr21 Pilot)| Sayeda Rehmat | Research Prototype"
    "</div>",
    unsafe_allow_html=True
)
