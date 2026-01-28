# Databricks notebook source
# MAGIC %md
# MAGIC # 04_Model_Training_Evaluation
# MAGIC
# MAGIC Trains and evaluates a classification model to estimate loan default risk. This notebook covers data loading, model training, evaluation, business logic conversion, and handling class imbalance.

# COMMAND ----------

# MAGIC %md
# MAGIC ### MLflow Setup
# MAGIC
# MAGIC This section initializes MLflow for experiment tracking. MLflow will log model parameters, metrics, and artifacts, enabling reproducibility and comparison across different model runs.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE VOLUME IF NOT EXISTS workspace.home_credit.mlflow_tmp;
# MAGIC

# COMMAND ----------

import os

os.environ["MLFLOW_DFS_TMP"] = "/Volumes/workspace/home_credit/mlflow_tmp"


# COMMAND ----------

# MAGIC %md
# MAGIC ### Import Required Libraries
# MAGIC
# MAGIC Import essential Python libraries for data manipulation, visualization, and model evaluation. These include pandas, matplotlib, and scikit-learn metrics for plotting and assessing model performance.

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay


# COMMAND ----------

# MAGIC %md
# MAGIC ### Set MLflow Experiment
# MAGIC
# MAGIC Configure the MLflow experiment path to organize and track all model runs related to the Home Credit risk baseline models.

# COMMAND ----------

# DBTITLE 1,Untitled
import mlflow
import mlflow.spark

mlflow.set_experiment("/Users/aishwaryaprabha29@gmail.com/home_credit_risk_baseline_models")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Gold Features Table
# MAGIC
# MAGIC Load the ML-ready Gold table containing engineered features into a Spark DataFrame. This dataset will be used for training and evaluating classification models.

# COMMAND ----------

# DBTITLE 1,Untitled
gold_ready_df = spark.table("workspace.home_credit.gold_application_features")



# COMMAND ----------

# MAGIC %md
# MAGIC ### Train-Test Split
# MAGIC
# MAGIC Split the Gold features data into training and test sets using an 80/20 ratio. This ensures unbiased evaluation of model performance on unseen data.

# COMMAND ----------

# DBTITLE 1,Untitled
train_df, test_df = gold_ready_df.randomSplit([0.8, 0.2], seed=42)




# COMMAND ----------

# MAGIC %md
# MAGIC ### Logistic Regression Model Training & Evaluation
# MAGIC
# MAGIC Train a baseline logistic regression model to predict loan default risk. Evaluate its performance using metrics such as ROC AUC, accuracy, precision, and recall. Log results and artifacts to MLflow for experiment tracking.

# COMMAND ----------

# DBTITLE 1,Untitled
from pyspark.sql.functions import col
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

with mlflow.start_run(run_name="Logistic_Regression_v2_with_model_logging"):

    # Model
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="TARGET"
    )

    lr_model = lr.fit(train_df)

    # Predictions
    predictions = lr_model.transform(test_df)

    # Convert to Pandas for plotting
    pdf = predictions.select(
        "TARGET",
        "probability"
    ).toPandas()

    # Extract probability of default (class = 1)
    pdf["prob_default"] = pdf["probability"].apply(lambda x: float(x[1]))


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

    # Precision
    precision = MulticlassClassificationEvaluator(
        labelCol="TARGET",
        predictionCol="prediction",
        metricName="weightedPrecision"
    ).evaluate(predictions)

    # Recall
    recall = MulticlassClassificationEvaluator(
        labelCol="TARGET",
        predictionCol="prediction",
        metricName="weightedRecall"
    ).evaluate(predictions)

    # Log parameters
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("features_col", "features")

    # Log metrics
    mlflow.log_metric("roc_auc", auc)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    fpr, tpr, _ = roc_curve(pdf["TARGET"], pdf["prob_default"])

    plt.figure()
    plt.plot(fpr, tpr, label="Logistic Regression")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Logistic Regression")
    plt.legend()

    roc_path = "/tmp/roc_curve.png"
    plt.savefig(roc_path)
    plt.close()

    # Log to MLflow
    mlflow.log_artifact(roc_path)
    
    cm = confusion_matrix(pdf["TARGET"], pdf["prob_default"] > 0.5)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    cm_path = "/tmp/confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    # Log to MLflow
    mlflow.log_artifact(cm_path)

    mlflow.spark.log_model(
    lr_model,
    artifact_path="model"
)




# COMMAND ----------

# MAGIC %md
# MAGIC ### Decision Tree Model Training & Evaluation
# MAGIC
# MAGIC Train a decision tree classifier as a benchmark model. Evaluate its performance and log metrics and artifacts to MLflow for comparison with the logistic regression baseline.

# COMMAND ----------

# DBTITLE 1,Untitled
from pyspark.ml.classification import DecisionTreeClassifier

with mlflow.start_run(run_name="Decision_Tree_Benchmark_with_model_logging"):

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

    precision = MulticlassClassificationEvaluator(
        labelCol="TARGET",
        predictionCol="prediction",
        metricName="weightedPrecision"
    ).evaluate(dt_predictions)

    recall = MulticlassClassificationEvaluator(
        labelCol="TARGET",
        predictionCol="prediction",
        metricName="weightedRecall"
    ).evaluate(dt_predictions)

    mlflow.log_param("model_type", "DecisionTree")
    mlflow.log_param("max_depth", 5)

    mlflow.log_metric("roc_auc", auc)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)


    print("DT AUC:", auc)
    print("DT Accuracy:", accuracy)

    mlflow.spark.log_model(
    dt_model,
    artifact_path="model"
)

