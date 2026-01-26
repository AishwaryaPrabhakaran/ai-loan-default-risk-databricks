# AI-Driven Loan Default Risk Prediction System

**Sponsor:** Databricks  
**Organisers:** Indian Data Club (IDC) & Codebasics  

**Hashtags:**  
#ResumeProjectChallenge #DatabricksWithIDC #Codebasics  

---

## Problem Statement

Loan approval is a critical decision for banks, directly impacting profitability, risk exposure, and customer experience.  
Traditional rule-based credit approval systems rely on fixed thresholds (such as income limits, credit scores, or employment type) and often treat large groups of customers in the same way.

This approach creates two major challenges:

### Financial Loss from Defaults
Banks approve loans for applicants who appear eligible under simple rules but later default due to complex risk patterns that static rules fail to capture.

### Missed Revenue and Operational Inefficiency
Low-risk customers are frequently subjected to the same manual reviews as higher-risk applicants, slowing down approvals, increasing operational costs, and reducing customer satisfaction.

---

## Why an AI-Driven System Is Needed

Loan default risk depends on multiple interacting factors such as income, credit amount, employment stability, family structure, and repayment burden. These relationships are often non-linear and evolve over time, making static rule-based systems insufficient.

An AI-driven system can:
- Learn risk patterns from historical loan outcomes
- Estimate the probability of default for each applicant
- Rank customers by risk instead of treating them uniformly
- Adapt more effectively than fixed rules as customer behavior changes

---

## Objective

The objective of this project is to build an **AI-driven loan default risk prediction system** that:
- Predicts the probability of loan default
- Segments applicants into risk buckets (Low, Medium, High)
- Converts predictions into **actionable credit decisions** such as fast approval, manual review, or rejection

This enables banks to reduce default-related losses while improving approval speed and operational efficiency.

---

## Data Overview

- **Dataset:** Home Credit Default Risk (Kaggle)  
- **Primary Table:** `application_train.csv`  

The dataset represents real-world loan application data used to assess default risk.

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

- **Model:** Logistic Regression (Spark ML)  
- **Evaluation Metrics:**  
  - AUC-ROC (primary)  
  - Accuracy (secondary)  

**Results:**
- AUC ≈ **0.61**  
- Accuracy ≈ **0.92**

Due to significant class imbalance in loan default data, AUC is the preferred metric as it better reflects the model’s ability to rank risky customers.

---

## AI Decision System (Beyond Just Training)

Rather than stopping at predictions, this project implements an AI-driven decision layer.

Model outputs are converted into:
- Default probability scores
- **Adaptive, quantile-based risk buckets**
- Recommended credit actions

**Observed default rates by risk bucket:**
- Low Risk ≈ **6%**  
- Medium Risk ≈ **11%**  
- High Risk ≈ **14%**

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

## Challenge Requirements Compliance

This project was developed as part of the **Codebasics Resume Project Challenge**, sponsored by **Databricks** and organised by **Indian Data Club (IDC)** and **Codebasics**.

The project complies with the challenge requirements by:
- Defining a self-directed, real-world problem statement
- Selecting a relevant public dataset
- Building a complete end-to-end solution using Databricks
- Creating a GitHub repository with clean, modular code
- Providing structured documentation and explainability
- Preparing the solution to be presentation-ready for a minimum 10-minute walkthrough

