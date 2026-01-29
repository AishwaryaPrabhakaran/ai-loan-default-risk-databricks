# System Architecture Diagram

This folder contains architecture diagrams for the AI-Driven Loan Default Risk Decision System.

The diagram illustrates the complete end-to-end flow of the system, from raw data ingestion to business decision outcomes.

## What the Diagram Shows

- Ingestion of the Home Credit Default Risk dataset
- Medallion architecture on Databricks (Bronze → Silver → Gold)
- Feature engineering and model training using Spark ML
- Logistic Regression model producing default probability scores
- AI-driven decision logic converting probabilities into:
  - Fast approval (Low Risk)
  - Manual review (Medium Risk)
  - Reject or collateral requirement (High Risk)

## How This Fits the Project

This diagram complements the notebooks in the `notebooks/` folder and provides a visual overview of how data, models, and decisions are connected in a production-oriented ML system.

