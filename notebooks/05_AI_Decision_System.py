# Databricks notebook source
# MAGIC %md
# MAGIC # 05_AI_Decision_System
# MAGIC
# MAGIC This notebook represents the AI-driven decision layer of the system. It converts model predictions into operational insights and actions that a bank’s credit team can directly use.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load Final Risk Decisions
# MAGIC
# MAGIC Read the stored table containing model predictions and business logic outputs for further analysis and operationalization.

# COMMAND ----------

#reading from a stored table
decision_df = spark.table("workspace.home_credit.final_risk_decisions")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Adaptive Risk Bucketing
# MAGIC
# MAGIC Instead of arbitrary thresholds, percentile-based risk bucketing is used so that risk categories adapt to the model score distribution.

# COMMAND ----------

quantiles = decision_df.approxQuantile(
    "default_probability",
    [0.7, 0.9],
    0.01
)

low_cutoff, high_cutoff = quantiles

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Assign Risk Buckets
# MAGIC
# MAGIC Assign each applicant to a risk bucket (Low, Medium, High) based on their default probability and calculated cutoffs.

# COMMAND ----------

from pyspark.sql.functions import col, when
decision_df = decision_df.withColumn(
    "risk_bucket",
     when(col("default_probability") >= high_cutoff, "High")
    .when(col("default_probability") >= low_cutoff, "Medium")
    .otherwise("Low")
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Risk Distribution Analysis
# MAGIC
# MAGIC Analyze the distribution of applicants across risk buckets to understand portfolio segmentation.

# COMMAND ----------

#Risk distribution
decision_df.groupBy("risk_bucket").count().show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Action Distribution Analysis
# MAGIC
# MAGIC Review the distribution of recommended actions to ensure operational alignment with business strategy.

# COMMAND ----------

# Action distribution
decision_df.groupBy("recommended_action").count().show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Bucket Default Rate Analysis
# MAGIC
# MAGIC Calculate default rates for each risk bucket to validate the effectiveness of risk segmentation.

# COMMAND ----------

from pyspark.sql.functions import col,lit, sum as spark_sum

bucket_stats = (
    decision_df
    .groupBy("risk_bucket")
    .agg(
        spark_sum(col("TARGET")).alias("defaults"),
        spark_sum((1 - col("TARGET"))).alias("non_defaults"),
        spark_sum(lit(1)).alias("total")
    )
    .withColumn(
        "default_rate",
        col("defaults") / col("total")
    )
)

bucket_stats.show(truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: AI Decision Logic
# MAGIC
# MAGIC Low Risk → Fast-track loan approval  
# MAGIC Medium Risk → Manual credit review  
# MAGIC High Risk → Reject or request collateral  
# MAGIC
# MAGIC Thresholds and actions are configurable based on business risk appetite.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Business Impact
# MAGIC
# MAGIC * Credit officers focus only on Medium-risk cases
# MAGIC * Risk team reduces default exposure
# MAGIC * Customers with Low risk get faster approvals
# MAGIC
# MAGIC This transforms raw predictions into real operational value.