# AI-Driven Loan Default Risk Prediction System

**Sponsor:** Databricks  
**Organisers:** Indian Data Club (IDC) & Codebasics  

**Hashtags:**  
#ResumeProjectChallenge #DatabricksWithIDC #Codebasics  

### This project demonstrates an end-to-end AI-driven credit risk decision system built on Databricks.  
### The focus is not only on model training, but on converting predictions into real, explainable business decisions.
---
## 📽️ Project Walkthrough

▶️ Video (10 mins, unlisted): https://youtu.be/V0JOPDGC_Xo  

## 📊 Presentation Deck

The presentation used in the walkthrough is available here:
- [Download Presentation (PDF)](presentation/AI_Loan_Default_Risk_Predictions_Presentation.pdf)

---
## Problem Statement

Loan approval is a critical decision for banks, directly affecting profitability, risk exposure, and customer experience.
Traditional rule-based credit approval systems rely on fixed thresholds such as income limits, credit scores, or employment type. These systems often treat large groups of applicants uniformly, ignoring complex risk patterns.

This approach creates two major challenges:

### Financial Loss from Defaults
Applicants who pass simple rules may still default due to hidden risk factors.

### Missed Revenue and Operational Inefficiency
Low-risk customers undergo unnecessary manual reviews, slowing approvals and increasing costs.

---

## Why an AI-Driven System Is Needed

Loan default risk depends on multiple interacting factors such as income, credit amount, employment stability, family structure, and repayment burden. These relationships are often non-linear and evolve over time.

An AI-driven system can:
- Learn risk patterns from historical loan outcomes
- Estimate default probability for each applicant
- Rank customers by relative risk instead of treating all applicants equally
- Adapt better than fixed rules as customer behavior changes
---

## Objective

The objective of this project is to build an **AI-driven loan default risk prediction system** that:
- Predicts the probability of loan default
- Segments applicants into risk buckets (Low, Medium, High)
- Converts predictions into **actionable credit decisions** such as fast approval, manual review, or rejection

This enables banks to reduce default-related losses while improving approval speed and operational efficiency.

---

## Dataset Overview

- **Dataset:** Home Credit Default Risk
- **Source:** Kaggle  
  https://www.kaggle.com/competitions/home-credit-default-risk/data

The dataset contains real-world loan application records used to predict whether a borrower will default on a loan. It includes applicant demographics, financial attributes, and credit-related information.

The data is highly imbalanced, which makes it well-suited for demonstrating:
- Risk ranking instead of binary classification
- Business-aligned evaluation using ROC-AUC
- Probability-based decision systems
---

## Architecture (Medallion Pattern)

The solution is built using Databricks and follows the Medallion Architecture:

- **Bronze:** Raw data ingestion into Unity Catalog tables  
- **Silver:** Business-driven data cleaning and validation  
- **Gold:** ML-ready feature generation using Spark ML pipelines  

This structure ensures data quality, traceability, and scalability.

---

## Feature Engineering

Key features used in the model include:
- Income and credit amounts
- Employment type and education
- Family and housing information
- **Credit-to-Income Ratio** (engineered feature)

Missing numeric values are handled using **median imputation**, which is robust to outliers commonly present in financial data.

---

## Model & Evaluation

**Model:** Logistic Regression (Spark ML)

**Evaluation Metrics:**
- ROC-AUC (primary)
- Accuracy (secondary)

**Results:**
- ROC-AUC ≈ 0.61  
- Accuracy ≈ 0.92  

Due to significant class imbalance in loan default data, accuracy is misleading.  
ROC-AUC is preferred as it reflects the model’s ability to **rank risky applicants**, which aligns with business needs.

Model experiments, metrics, and evaluation artifacts (ROC Curve and Confusion Matrix) were tracked using **MLflow**.

---

## AI Decision System (Beyond Just Training)

Instead of stopping at predictions, this project implements a decision layer.

Model outputs are converted into:
- Default probability scores
- Quantile-based risk buckets (Low, Medium, High)
- Recommended credit actions

**Indicative default rates by risk bucket:**
- Low Risk ≈ 6%
- Medium Risk ≈ 11%
- High Risk ≈ 14%

This demonstrates effective risk separation and supports operational decision-making.

---

## Business Impact

- Faster approvals for low-risk applicants  
- Focused manual reviews for medium-risk cases  
- Reduced default exposure and improved risk control  

The system enables banks to allocate effort where it matters most while improving customer experience.

---

## End-to-End Workflow

**Data → Tables → Features → Model → Decisions → Business Actions**

This project demonstrates a complete AI system lifecycle rather than isolated model training.

---

## Limitations & Next Steps

- The model is a baseline and can be improved using richer credit history data
- Risk thresholds and actions are configurable based on business risk appetite
- More advanced models can be explored once explainability requirements are met

---

## Repository Overview

- `notebooks/` – Model training and evaluation code  
- `mlflow_artifacts/` – ROC curve and confusion matrix  
- `presentation/` – Final presentation deck (PDF)  
- `README.md` – Project documentation
---

## Challenge Requirements Compliance
This project was developed as part of the **Codebasics Resume Project Challenge**, sponsored by **Databricks** and organised by **Indian Data Club (IDC)** and **Codebasics**.

The project complies with the challenge requirements by:
- Defining a self-directed, real-world problem statement
- Selecting a relevant public dataset
- Building a complete end-to-end solution using Databricks
- Creating a GitHub repository with clean, modular code
- Providing structured documentation and explainability
- Preparing the solution to be presentation-ready for a minimum 10-minute walkthrough

