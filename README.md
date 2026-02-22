# Mapping Shared Risk Profiles of Multimorbidity and Functional Decline in U.S. Adults

[![R](https://img.shields.io/badge/R-4.5.2-blue.svg)](https://www.r-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Overview

This repository contains the complete R analysis code for the paper:

>Okeagu PN, Okeke G, Odu EC, Kebede YT. Mapping Shared Risk Profiles of Multimorbidity and Functional Decline Among U.S. Adults: A Supervised Machine Learning Analysis. 2026.

We used machine learning to identify shared and distinct risk clusters for multimorbidity and functional decline in 2,154 participants from the MIDUS Refresher 2 study (ages 25-74).

Key Findings

- Random Forest achieved the highest performance (multimorbidity AUC=0.848; functional decline AUC=0.866)
- Four risk clusters identified: demographic (age), anthropometric (BMI, waist), socioeconomic (income, education), and psychosocial-behavioral (stress, sleep, mental health)
- Bidirectional relationship: each outcome strongly predicts the other
- Modifiable targets: BMI, sleep quality, and perceived stress are key drivers of functional decline

Repository Structure


---

## 🔧 Requirements

### R Environment
- R version 4.5.2 or higher
- RStudio (recommended) or any R environment

### Required R Packages
```r
# Core data science
tidyverse     # Data manipulation and visualization (v2.0.0)
haven         # SPSS data import (v2.5.4)
janitor       # Data cleaning (v2.2.0)
labelled      # Variable labels (v2.12.0)

# Machine Learning
caret         # Model training framework (v6.0-94)
glmnet        # LASSO regression (v4.1-8)
randomForest  # Random Forest models (v4.7-1.1)
xgboost       # XGBoost models (v1.7.5.1)

# Evaluation & Interpretation
pROC          # ROC curve analysis (v1.18.5)
SHAPforxgboost # SHAP values (v0.1.1)

# Visualization
ggplot2       # Figures (v3.4.4)
gridExtra     # Multi-panel figures (v2.3)
knitr         # Report generation (v1.45)

# Missing Data
missForest    # Missing data imputation (v1.5)

