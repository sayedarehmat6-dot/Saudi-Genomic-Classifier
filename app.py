import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(
    page_title="SGC | Saudi Genomic Classifier",
    page_icon="🧬",
    layout="wide" # Wide mode for a better dashboard feel
)

# 2. Professional Model Loader
@st.cache_resource 
def load_xgb_model():
    model_path = os.path.join("models", "pathogenicity_model.json")
    if not os.path.exists(model_path):
        return None
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model

model = load_xgb_model()

# 3. Sidebar - Personal Branding & Info
with st.sidebar:
    st.title("🧬 SGC v1.0")
    st.markdown(f"""
    **Developer:** Sayeda Rehmat
    
    **Project Overview:**
    The Saudi Genomic Classifier (SGC) is a specialized AI tool designed to predict the pathogenicity of genetic variants within the Saudi Arabian population.
    
    **Technical Stack:**
    * **Data:** Integrated PAVS + gnomAD
    * **Model:** Optimized XGBoost
    * **Focus:** Chromosome 21 (Pilot)
    """)
    st.divider()
    st.caption("Decision Support Tool for Genomic Research")

# 4. Main UI Header
st.title(" Saudi Genomic Classifier Dashboard")
st.markdown("---")

# 5. Input Section
st.subheader(" Variant Parameters")
with st.container():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pos = st.number_input("Genomic Position (BP)", value=10000000)
        impact = st.selectbox("VEP Functional Impact", ["HIGH", "MODERATE", "LOW", "MODIFIER"])
    
    with col2:
        global_af = st.number_input("Global Allele Frequency", format="%.6f", value=0.0001)
        pli = st.slider("pLI Score", 0.0, 1.0, 0.5)
        
    with col3:
        loeuf = st.slider("LOEUF Score", 0.0, 2.0, 0.5)

# Mapping impact
impact_map = {'HIGH': 3, 'MODERATE': 2, 'LOW': 1, 'MODIFIER': 0}
impact_val = impact_map[impact]

st.markdown("---")

# 6. Prediction & Dashboard Logic
if st.button(" Execute Diagnostic Analysis", use_container_width=True):
    if model is not None:
        input_data = np.array([[pos, global_af, impact_val, pli, loeuf]])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        # Results Dashboard
        st.subheader("📊 Analysis Results")
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            if prediction == 1:
                st.error("### CLASS: PATHOGENIC")
                st.metric("Confidence", f"{probability[1]*100:.2f}%")
            else:
                st.success("### CLASS: BENIGN / VUS")
                st.metric("Confidence", f"{probability[0]*100:.2f}%")
            
            st.write("**Summary:** The model analyzed the interaction between allelic rarity and gene constraint to determine clinical significance.")

        with res_col2:
            # Professional Bar Chart for Probabilities
            fig = go.Figure(go.Bar(
                x=[probability[0], probability[1]],
                y=['Benign', 'Pathogenic'],
                orientation='h',
                marker_color=['#2ecc71', '#e74c3c']
            ))
            fig.update_layout(
                title="Prediction Probability Distribution",
                xaxis_title="Confidence Level",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            
    else:
        st.error("Configuration Error: Model file missing in /models directory.")

 # 7. Professional Footer (Fixed)
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Saudi Genomic Classifier | Developed by Sayeda Rehmat | Precision Medicine Initiative"
    "</div>", 
    unsafe_allow_html=True # Removed the '_stdio' error
)
