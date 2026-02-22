# Mapping Shared Risk Profiles of Multimorbidity and Functional Decline in U.S. Adults

[![R](https://img.shields.io/badge/R-4.4.0-blue.svg)](https://www.r-project.org/)
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
