# Databricks notebook source
# MAGIC %md
# MAGIC # 02_Silver_Cleaning
# MAGIC
# MAGIC This notebook applies business-driven data cleaning rules to prepare trusted analytical data for downstream use. The workflow includes loading Bronze data, handling missing values, capping outliers, and saving the cleaned Silver table.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Import Required Libraries
# MAGIC
# MAGIC Import necessary PySpark functions for data cleaning and transformation.

# COMMAND ----------

from pyspark.sql.functions import col, when, lit
from pyspark.sql.functions import percentile_approx


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Load Bronze Table
# MAGIC
# MAGIC Read the raw application data from the Bronze table (`workspace.home_credit.bronze_application_train`) into a DataFrame for cleaning.

# COMMAND ----------

bronze_df = spark.table("workspace.home_credit.bronze_application_train")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Drop Rows with Critical Nulls
# MAGIC
# MAGIC Remove rows with missing values in key columns (`TARGET`, `AMT_INCOME_TOTAL`) to ensure data integrity.

# COMMAND ----------

# Drop rows with critical missing values
silver_df = bronze_df.filter(
    col("TARGET").isNotNull() &
    col("AMT_INCOME_TOTAL").isNotNull()
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Handle Categorical Missing Values
# MAGIC
# MAGIC Replace nulls in important categorical columns with 'Unknown' to maintain consistency.

# COMMAND ----------

# Handle categorical missing values
categorical_cols = [
    "NAME_TYPE_SUITE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE"
]

for c in categorical_cols:
    silver_df = silver_df.withColumn(
        c,
        when(col(c).isNull(), lit("Unknown")).otherwise(col(c))
    )

# COMMAND ----------

# DBTITLE 1,Cell 10
# MAGIC %md
# MAGIC ## Step 5: Cap Extreme Values
# MAGIC
# MAGIC Limit extreme values in `AMT_INCOME_TOTAL` and `AMT_CREDIT` at the 99th percentile to reduce the impact of outliers.

# COMMAND ----------

# Cap extreme values (outliers)
# Cap at 99th percentile (bank-safe logic)
income_cap = silver_df.select(
    percentile_approx("AMT_INCOME_TOTAL", 0.99)
).first()[0]

credit_cap = silver_df.select(
    percentile_approx("AMT_CREDIT", 0.99)
).first()[0]

# COMMAND ----------

silver_df = silver_df.withColumn(
    "AMT_INCOME_TOTAL",
    when(col("AMT_INCOME_TOTAL") > income_cap, income_cap)
    .otherwise(col("AMT_INCOME_TOTAL"))
)

silver_df = silver_df.withColumn(
    "AMT_CREDIT",
    when(col("AMT_CREDIT") > credit_cap, credit_cap)
    .otherwise(col("AMT_CREDIT"))
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Save Cleaned Data
# MAGIC
# MAGIC Persist the cleaned DataFrame as a Silver table (`workspace.home_credit.silver_application_cleaned`) for trusted analytics.

# COMMAND ----------

silver_df.write.mode("overwrite").saveAsTable(
    "workspace.home_credit.silver_application_cleaned"
)


# COMMAND ----------

# DBTITLE 1,Cell 15
# MAGIC %md
# MAGIC ## Step 7: Inspect Cleaned Data
# MAGIC
# MAGIC Verify the row count and target distribution in the Silver table to confirm successful cleaning.

# COMMAND ----------

silver_df.count()
silver_df.select("TARGET").groupBy("TARGET").count()


# COMMAND ----------

display(silver_df.groupBy("TARGET").count())
