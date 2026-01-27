# Databricks notebook source
# MAGIC %md
# MAGIC # 04_Model_Training_Evaluation
# MAGIC
# MAGIC Trains and evaluates a classification model to estimate loan default risk. This notebook covers data loading, model training, evaluation, business logic conversion, and handling class imbalance.

# COMMAND ----------

# DBTITLE 1,Cell 2
# MAGIC %md
# MAGIC ## Step 0: MLflow Setup
# MAGIC
# MAGIC Set up MLflow for experiment tracking. This enables logging of model parameters, metrics, and artifacts for reproducibility and comparison.

# COMMAND ----------

# DBTITLE 1,Untitled
import mlflow
import mlflow.spark

mlflow.set_experiment("/Users/aishwaryaprabha29@gmail.com/home_credit_risk_baseline_models")


# COMMAND ----------

# DBTITLE 1,Cell 4
# MAGIC %md
# MAGIC ## Step 1: Load Gold Features
# MAGIC
# MAGIC Load the ML-ready Gold table into a Spark DataFrame for model training and evaluation.

# COMMAND ----------

# DBTITLE 1,Untitled
gold_ready_df = spark.table("workspace.home_credit.gold_application_features")



# COMMAND ----------

# DBTITLE 1,Cell 6
# MAGIC %md
# MAGIC ## Step 2: Train-Test Split
# MAGIC
# MAGIC Split the data into training and test sets to enable unbiased model evaluation.

# COMMAND ----------

# DBTITLE 1,Untitled
train_df, test_df = gold_ready_df.randomSplit([0.8, 0.2], seed=42)




# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Step 3:  Logistic Regression Model Training & Evaluation
# MAGIC
# MAGIC Train a logistic regression model as a baseline model, evaluate its performance, and log results to MLflow for comparison.

# COMMAND ----------

# DBTITLE 1,Untitled
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

with mlflow.start_run(run_name="Logistic_Regression_Baseline"):

    # Model
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="TARGET"
    )

    lr_model = lr.fit(train_df)

    # Predictions
    predictions = lr_model.transform(test_df)

    # Evaluation
    auc_evaluator = BinaryClassificationEvaluator(
        labelCol="TARGET",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    auc = auc_evaluator.evaluate(predictions)

    accuracy = MulticlassClassificationEvaluator(
        labelCol="TARGET",
        predictionCol="prediction",
        metricName="accuracy"
    ).evaluate(predictions)

    # Log parameters
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("features_col", "features")

    # Log metrics
    mlflow.log_metric("roc_auc", auc)
    mlflow.log_metric("accuracy", accuracy)

    print("AUC:", auc)
    print("Accuracy:", accuracy)



# COMMAND ----------

# MAGIC %md
# MAGIC ###Step 4:  Decision Tree Model Training & Evaluation
# MAGIC
# MAGIC Train a decision tree classifier as a benchmark, evaluate its performance, and log results to MLflow for comparison.

# COMMAND ----------

# DBTITLE 1,Untitled
from pyspark.ml.classification import DecisionTreeClassifier

with mlflow.start_run(run_name="Decision_Tree_Benchmark"):

    dt = DecisionTreeClassifier(
        featuresCol="features",
        labelCol="TARGET",
        maxDepth=5
    )

    dt_model = dt.fit(train_df)
    dt_predictions = dt_model.transform(test_df)

    auc = auc_evaluator.evaluate(dt_predictions)
    accuracy = MulticlassClassificationEvaluator(
        labelCol="TARGET",
        predictionCol="prediction",
        metricName="accuracy"
    ).evaluate(dt_predictions)

    mlflow.log_param("model_type", "DecisionTree")
    mlflow.log_param("max_depth", 5)

    mlflow.log_metric("roc_auc", auc)
    mlflow.log_metric("accuracy", accuracy)

    print("DT AUC:", auc)
    print("DT Accuracy:", accuracy)
