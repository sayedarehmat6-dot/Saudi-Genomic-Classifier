# Saudi-Genomic-Classifier (SGC)
Bridging the Diversity Gap in Pathogenicity Prediction via Interpretable AI

🧬 Project Abstract

Existing genomic pathogenicity predictors are trained on a dataset that reflects the western representation, resulting in the formation of a "representation gap" that negatively affects their performance in the diagnosis of Saudi individuals. To solve this problem, Saudi Genomic Classifier (SGC) was designed as an addition to the existing genomic tools that incorporates Saudi Clinical Database and gnomAD/ClinVar features. Through the implementation of cost-sensitive XGBoost trained with the SMOTE technique, SGC delivers a classification layer for uniquely Saudi genomic variants.

🔬 Methodological 

In order to deliver high-quality output in line with computational biology standards, I employed the following methodology:


Data harmonization: Consolidation of multiple heterogeneous databases (PAVS, ClinVar, gnomAD) into a single master feature matrix comprising over 3,000 genomic variants.


Mitigation of class imbalance problem: Genomic data is typically imbalanced due to the nature of pathogenicity. In order to address the issue, I implemented the SMOTE (Synthetic Minority Over-sampling Technique) and adjusted the scale_pos_weight hyperparameter in XGBoost in order to improve the model's ability to recall rare deleterious alleles.


Feature engineering (biological): Gene-level intolerance measures (pLI and LOEUF), capture evolutionary constraints necessary to distinguish between VUS and genuine pathogens.

Model Explainability (XAI)

Clinical genetics cannot function with a 'Black Box' method; therefore, SHAP (SHapley Additive Explanations) is used to guarantee biological plausibility of model predictions:


1. Global Feature Importance:

The primary predictor of this model is gnomAD pLI. This finding is consistent with our understanding of the genes and the necessity of high intolerance to the consequences of loss-of-function for the detection of a pathogenic phenotype.


2. Local Feature Impact (Beeswarm Logic):

 The beeswarm chart also demonstrates the high-fidelity nature of this learning procedure:
 
 Increasing impact_score  correctly pulls the prediction towards pathogenic (Red dots right of the center).

 It effectively captures the cooperative impact of low global frequencies (global_af) and high gene intolerances.


Repository Structure


 data/
 
 SaudiVariantAIMasterDataset.csv # Pre-processed Master Feature Matrix
 models/
 
 pathogenicity_model.json # Serialized XGBoost Classifier
 
 images/
 
 shapsummarybar.png # Global importance ranking
 
 shap_beeswarm.png # Feature-level logic distribution
 
 SaudiGenomicClassifier.ipynb # Full Computational Pipeline


 Usage & Deployment

 The serialized model is provided in JSON format to ensure broadest compatibility with clinical workstations.



Python

import xgboost as xgb

model = xgb.XGBClassifier()

model.loadmodel("models/pathogenicitymodel.json")

Input: [pos, globalaf, impactscore, gnomadpli, gnomadloeuf]


This project is a proof-of-concept to enable precision medicine within the Kingdom. By ranking variants by their population specificity and through explainable AI, the  time of diagnoses can be shortened for the Saudi Arabian population, contributing to the overall aims of the Saudi Genome Program.
