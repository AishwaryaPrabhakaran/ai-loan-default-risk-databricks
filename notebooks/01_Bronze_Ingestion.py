# Databricks notebook source
# MAGIC %md
# MAGIC # 01_Bronze_Ingestion
# MAGIC
# MAGIC This notebook loads raw loan application data into Bronze tables for reproducible processing. The workflow includes reading the source CSV, saving it as a managed table, and inspecting the ingested data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load Raw Application Data
# MAGIC
# MAGIC Read the raw loan application data from CSV into a Spark DataFrame. The data is sourced from `/Volumes/workspace/home_credit/raw_data/application_train.csv`.

# COMMAND ----------


bronze_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv("/Volumes/workspace/home_credit/raw_data/application_train.csv")
)



# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Save Data as Bronze Table
# MAGIC
# MAGIC Persist the loaded DataFrame as a Bronze table in Unity Catalog for reproducible downstream processing. The table is named `workspace.home_credit.bronze_application_train`.

# COMMAND ----------

bronze_df.write.mode("overwrite").saveAsTable(
    "workspace.home_credit.bronze_application_train"
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Inspect Bronze Table
# MAGIC
# MAGIC Verify the ingested data by checking the row count and schema of the Bronze table.

# COMMAND ----------

bronze_df.count()
bronze_df.printSchema()
