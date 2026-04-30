# Saudi Genomic Classifier (SGC)

**Population-Aware Pathogenicity Prediction using Interpretable AI (Chr21 Pilot Study)**

---

## Overview

The **Saudi Genomic Classifier (SGC)** is a machine learning-based framework that explores **population-aware variant pathogenicity prediction**.

Most existing predictors are based on datasets that mainly include Western populations. This leads to a **representation bias** and can lower the reliability of diagnoses in underrepresented populations, such as those in Saudi Arabia.

This project fills that gap by combining **Saudi-specific genomic data** with global resources and using **interpretable machine learning**.

---

## Objective

* Develop a **pathogenicity prediction model** using biologically meaningful features.
* Integrate **local (Saudi) and global genomic datasets**.
* Show the feasibility of **population-aware prediction**.
* Create an **interactive tool** for variant analysis.

---

## Data Sources

This project brings together several genomic resources:

* **PAVS** – Saudi clinical variant dataset
* **ClinVar** – variant pathogenicity annotations
* **gnomAD** – population allele frequency and constraint metrics

These datasets were combined into a unified feature matrix.

---

## Methodology

### 1. Data Integration

Different datasets were merged into a **single master dataset (~3,000 variants)**.

---

### 2. Feature Engineering

The model uses biologically relevant features:

* `global_af` → allele frequency (rarity signal)
* `impact` → functional consequence (VEP-derived)
* `pli` → gene-level intolerance
* `loeuf` → loss-of-function constraint
* `pos` → genomic position (contextual feature)

---

### 3. Class Imbalance Handling

Genomic datasets often show high imbalance.

To address this:

* **SMOTE** was used to oversample the minority class.
* **scale_pos_weight** was adjusted in XGBoost.

---

### 4. Model

* **Algorithm:** XGBoost Classifier
* **Task:** Binary classification (Pathogenic vs Benign/VUS)
* **Features:** 5
* **Training Scope:** Chromosome 21

---

## Genomic Scope

This project is a **Chromosome 21 pilot study**.

This design:

* facilitates controlled experimentation
* validates the integration of Saudi plus global data
* offers a foundation for future genome-wide scaling

 This model is **not genome-wide**.

---

## Model Explainability (XAI)

To ensure clarity, **SHAP (SHapley Additive Explanations)** was used.

### Global Insights

* **pLI is a dominant predictor.**
* Genes with high constraint are more likely to be pathogenic.

### Local Behavior

* High impact variants skew predictions toward pathogenic.
* Rare variants along with high constraint increase the chance of being pathogenic.

---

## Streamlit Application
https://saudi-genomic-classifier-eprkvza6vp5jnom9zenzzb.streamlit.app/

An interactive dashboard is included  with:

* Variant-level prediction
* Dataset exploration
* Model insights (feature importance + SHAP)

 

## Repository Structure


data/
 └── Saudi_Variant_AI_Master_Dataset.csv

models/
 └── pathogenicity_model.json

images/
 ├── shap_summary_bar.png
 └── shap_beeswarm.png

Saudi_Genomic_Classifier.ipynb   # Full pipeline
app.py                           # Streamlit interface
requirements.txt
```



 
This is a **research prototype,** not a clinical diagnostic tool.

---

## Future Work

* Expand to genome-wide prediction.
* Add more features (CADD, REVEL, conservation scores).
* Validate with independent datasets.
* Enhance population-specific calibration.

---

## Conclusion

This project shows the feasibility of combining **population-specific genomic data** with **interpretable machine learning** for pathogenicity prediction.

It establishes a basis for reducing **representation bias** in genomic diagnostics and supports broader aims of **precision medicine initiatives**.

---

## Author

**Sayeda Rehmat**  
Bioinformatics Research Project

---

## Note

The complete data processing, feature engineering, and model training pipeline is documented in the accompanying Jupyter Notebook:

`Saudi_Genomic_Classifier.ipynb`
