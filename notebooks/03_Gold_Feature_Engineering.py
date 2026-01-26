# Databricks notebook source
# MAGIC %md
# MAGIC # 03_Gold_Feature_Engineering
# MAGIC
# MAGIC Transforms cleaned Silver data into ML-ready features using Spark ML pipelines. This notebook covers feature selection, feature engineering, categorical encoding, imputation, vector assembly, scaling, and saving the final Gold table.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load Cleaned Silver Data
# MAGIC
# MAGIC Read the trusted Silver table (`workspace.home_credit.silver_application_cleaned`) into a DataFrame for feature engineering.

# COMMAND ----------

from pyspark.sql.functions import col
silver_df = spark.table("workspace.home_credit.silver_application_cleaned")



# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Select Relevant Features
# MAGIC
# MAGIC Select key columns for modeling, including target, numeric, and categorical features.

# COMMAND ----------

gold_df = silver_df.select(
    "TARGET",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "CNT_FAM_MEMBERS",
    "NAME_INCOME_TYPE",
    "OCCUPATION_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE"
)



# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Feature Engineering
# MAGIC
# MAGIC Create new features such as `CREDIT_INCOME_RATIO` to enhance model performance.

# COMMAND ----------

gold_df = gold_df.withColumn(
    "CREDIT_INCOME_RATIO",
    col("AMT_CREDIT") / col("AMT_INCOME_TOTAL")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Import Spark ML Libraries
# MAGIC
# MAGIC Import necessary Spark ML classes for categorical encoding, imputation, vector assembly, and scaling.

# COMMAND ----------

# Spark ML Feature Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler
)
from pyspark.ml import Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Encode Categorical Features
# MAGIC
# MAGIC Apply StringIndexer and OneHotEncoder to categorical columns for ML compatibility.

# COMMAND ----------

#Categorical processing
categorical_cols = [
    "NAME_INCOME_TYPE",
    "OCCUPATION_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE"
]

indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=f"{c}_idx",
        handleInvalid="keep"
    )
    for c in categorical_cols
]

encoders = [
    OneHotEncoder(
        inputCol=f"{c}_idx",
        outputCol=f"{c}_ohe"
    )
    for c in categorical_cols
]


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Impute Missing Values
# MAGIC
# MAGIC Use Imputer to fill missing values in numeric columns.

# COMMAND ----------

from pyspark.sql.functions import col, sum as spark_sum

numeric_cols = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "CNT_FAM_MEMBERS",
    "CREDIT_INCOME_RATIO"
]

gold_df.select([
    spark_sum(col(c).isNull().cast("int")).alias(c)
    for c in numeric_cols
]).show()


# COMMAND ----------

from pyspark.ml.feature import Imputer


# COMMAND ----------

imputer = Imputer(
    inputCols=numeric_cols,
    outputCols=[f"{c}_imputed" for c in numeric_cols]
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Assemble Feature Vector
# MAGIC
# MAGIC Combine all processed features into a single vector for modeling.

# COMMAND ----------

# numerical features
assembler = VectorAssembler(
    inputCols=[f"{c}_ohe" for c in categorical_cols] +
              [f"{c}_imputed" for c in numeric_cols],
    outputCol="features_raw",
    handleInvalid="skip"   # extra safety
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Scale Features
# MAGIC
# MAGIC Standardize feature vectors using StandardScaler for improved model training.

# COMMAND ----------

scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features"
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Build and Apply ML Pipeline
# MAGIC
# MAGIC Create and fit a Spark ML pipeline to transform the data into ML-ready format.

# COMMAND ----------

#Build Pipeline
pipeline = Pipeline(
    stages=indexers + encoders + [imputer, assembler, scaler]
)

gold_model = pipeline.fit(gold_df)
gold_ready_df = gold_model.transform(gold_df)



# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10: Save Gold Table
# MAGIC
# MAGIC Persist the final ML-ready DataFrame as a Gold table (`workspace.home_credit.gold_application_features`).

# COMMAND ----------

gold_ready_df.select("TARGET", "features").write.mode("overwrite").saveAsTable(
    "workspace.home_credit.gold_application_features"
)
