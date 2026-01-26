# Databricks notebook source
# MAGIC %md
# MAGIC # 04_Model_Training_Evaluation
# MAGIC
# MAGIC Trains and evaluates a classification model to estimate loan default risk. This notebook covers data loading, model training, evaluation, business logic conversion, and handling class imbalance.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load Gold Features
# MAGIC
# MAGIC Read the ML-ready Gold table (`workspace.home_credit.gold_application_features`) into a DataFrame for model training.

# COMMAND ----------

gold_ready_df = spark.table("workspace.home_credit.gold_application_features")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Train-Test Split
# MAGIC
# MAGIC Split the data into training and test sets for model development and evaluation.

# COMMAND ----------

train_df, test_df = gold_ready_df.randomSplit([0.8, 0.2], seed=42)




# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Train Logistic Regression Model
# MAGIC
# MAGIC Fit a logistic regression model to predict loan default risk using Spark ML.

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(
    featuresCol="features",
    labelCol="TARGET"
)

lr_model = lr.fit(train_df)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Generate Predictions
# MAGIC
# MAGIC Apply the trained model to the test set to generate predictions.

# COMMAND ----------

predictions = lr_model.transform(test_df)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Model Evaluation
# MAGIC
# MAGIC Evaluate model performance using AUC and accuracy metrics.

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

auc_evaluator = BinaryClassificationEvaluator(
    labelCol="TARGET",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

auc = auc_evaluator.evaluate(predictions)
print("AUC:", auc)


# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

accuracy = MulticlassClassificationEvaluator(
    labelCol="TARGET",
    predictionCol="prediction",
    metricName="accuracy"
).evaluate(predictions)

print("Accuracy:", accuracy)
