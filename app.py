import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
import os

# 1. Page Configuration
st.set_page_config(
    page_title="SGC: Saudi Genomic Classifier",
    page_icon="🧬",
    layout="centered"
)

# 2. Professional Model Loader
@st.cache_resource 
def load_xgb_model():
    # Define the path to your model file
    model_path = os.path.join("models", "pathogenicity_model.json")
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please check your folder structure!")
        return None
        
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model

model = load_xgb_model()

# 3. Sidebar - Academic Context
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/4/47/KAUST_logo.png", width=150)
    st.title("About SGC")
    st.info("""
    **Developer:** [Your Name]
    **Lab Alignment:** Professor Xin Gao (KAUST)
    **Dataset:** PAVS + gnomAD
    **Scope:** Pilot study on Chromosome 21
    """)
    st.divider()
    st.write("This tool uses **Explainable AI (XAI)** to assist in clinical variant prioritization.")

# 4. Main UI
st.title("🇸🇦 Saudi Genomic Classifier")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Variant Input")
    pos = st.number_input("Genomic Position (BP)", value=10000000, help="Coordinate on Chromosome 21")
    global_af = st.number_input("Global Allele Frequency", format="%.6f", value=0.0001, help="Frequency from gnomAD")
    impact = st.selectbox("VEP Functional Impact", ["HIGH", "MODERATE", "LOW", "MODIFIER"])
    
    # Mapping impact to the integer scale used in XGBoost training
    impact_map = {'HIGH': 3, 'MODERATE': 2, 'LOW': 1, 'MODIFIER': 0}
    impact_val = impact_map[impact]

with col2:
    st.subheader("Constraint Scores")
    pli = st.slider("pLI Score", 0.0, 1.0, 0.5, help="Probability of intolerance to Loss-of-Function")
    loeuf = st.slider("LOEUF Score", 0.0, 2.0, 0.5, help="Lower scores indicate stronger selection against mutation")

st.divider()

# 5. Prediction Logic
if st.button("🚀 Run Pathogenicity Analysis", use_container_width=True):
    if model is not None:
        # Features must be in order: [pos, global_af, impact_score, gnomad_pli, gnomad_loeuf]
        input_data = np.array([[pos, global_af, impact_val, pli, loeuf]])
        
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        # 6. Professional Result Display
        st.subheader("Classification Result")
        
        if prediction == 1:
            st.error(f"### Result: **Likely PATHOGENIC**")
            st.metric("AI Confidence Score", f"{probability[1]*100:.2f}%")
            st.warning("Clinical Correlation Required: High probability of deleterious functional impact.")
        else:
            st.success(f"### Result: **Likely BENIGN / VUS**")
            st.metric("AI Confidence Score", f"{probability[0]*100:.2f}%")
            st.info("The variant does not meet the pathogenicity threshold based on current training data.")
            
    else:
        st.error("Model failed to load. Ensure 'pathogenicity_model.json' is inside the 'models' folder.")

# 7. Footer
st.markdown("---")
st.caption("© 2024 Saudi Genomic Classifier Pilot | Research Submission for KAUST Bioinformatics")
