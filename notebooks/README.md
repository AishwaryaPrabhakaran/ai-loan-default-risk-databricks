# Databricks Notebooks

This folder contains Databricks notebooks exported as Python source files.
Together, they implement the complete end-to-end pipeline for the
AI-Driven Loan Default Risk Decision System.

## Execution Order

The notebooks are designed to be executed sequentially:

1. **01_Bronze_Ingestion.py**  
   Ingests raw loan application data into Bronze tables in Unity Catalog.

2. **02_Silver_Cleaning.py**  
   Applies business-driven data cleaning, validation, and standardization.

3. **03_Gold_Feature_Engineering.py**  
   Generates ML-ready features and builds Spark ML pipelines.

4. **04_Model_Training_Evaluation.py**  
   Trains models, evaluates performance, and tracks experiments using MLflow.

5. **05_AI_Decision_System.py**  
   Converts model probabilities into percentile-based risk buckets and
   actionable credit decisions.

Each notebook produces outputs that are consumed by the next stage,
following the Medallion (Bronze → Silver → Gold) architecture.

