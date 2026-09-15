"""
Predicting Diabetes Risk with PySpark SQL and MLlib

---------------------------------------------------------

You've learned how to select, transform, filter, aggregate, and join
DataFrames with PySpark. Now you'll use these skills to turn a messy
patient dataset into something MLlib can train a model on.

The dataset below is randomly generated for this exercise and is
not real medical data, it's just structured to behave like real-world
health data usually does: mixed types, missing values, and inconsistent 
categories.

How this activity works
------------------------
1. Run this script. It will fail partway through (an exception about a missing 
   column) because the data isn't ready yet (this is expected)
2. Complete  the seven TODOs below using select / withColumn / filter /
   groupBy+agg / join etc.
3. Re-run the script. Once all TODOs are done, it should run end-to-end
   and print a trained logistic regression model with an AUC and accuracy
   score.
4. Try the "go further" suggestions at the bottom.

New function that will be helpful: to change a column's data type inside
withColumn, use .cast("double"), e.g. F.col("age").cast("double").
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

spark = SparkSession.builder \
    .appName("DiabetesRiskActivity") \
    .master("local[*]") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .config("spark.driver.host", "127.0.0.1") \
    .config("spark.sql.ansi.enabled", "false") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()


# ---------------------------------------------------------------------------
# STEP 0: Read in data
# ---------------------------------------------------------------------------

path = 'diabetes_data.csv'

patients_raw = spark.read.option("delimiter", ",").option("header", True).option("inferSchema", True).csv(path)

print("Raw patient data (note the messy types & missing values)")
patients_raw.show(10, truncate=False)
print(patients_raw.dtypes)

# A small lookup table you'll use later
smoker_lookup = spark.createDataFrame(
    [("Yes", 1.4), ("No", 1.0)],
    ["smoker", "smoker_risk_multiplier"]
)

# ---------------------------------------------------------------------------
# YOUR TASK: clean and reshape the data so it's ready for MLlib.
# Use select, withColumn, filter, groupBy/agg, and join.
# ---------------------------------------------------------------------------

# TODO 1: height_cm and weight_kg were loaded as STRINGS, and some patients
# have "unknown" instead of a number. Use withColumn to overwrite each column
# with F.col(...).cast("double"). 
patients_clean = patients_raw  # <-- build on this

# TODO 2: Use .filter(...) to drop any row where height_cm or weight_kg is
# now NULL (hint: F.col("height_cm").isNotNull()), and also drop any row where
# glucose_mgdl == -1 (a dummy value indicating "not measured").


# TODO 3: Add a "bmi" column with withColumn.
#   BMI = weight_kg / (height_m ** 2),  where height_m = height_cm / 100


# TODO 4: Add a "bp_category" column using F.when(...).otherwise(...):
#   - "High"     if bp_systolic >= 140 OR bp_diastolic >= 90
#   - "Elevated" if bp_systolic >= 130 (and doesn't already qualify as High)
#   - "Normal"   otherwise
# (Hint: chain a second .when(...) before .otherwise(...))


# TODO 5 (JOIN): Join patients_clean with smoker_lookup on the "smoker"
# column using how="inner". This attaches a smoker_risk_multiplier column.
# As a side effect of the inner join, it also drops any patient
# whose smoker status was "Unknown", since smoker_lookup only has "Yes"/"No".


# TODO 6 (Optional, but recommended): Use groupBy("bp_category").agg(...)
# to print the average glucose_mgdl and average bmi for each blood-pressure
# category. This won't feed into the model, but it's good practice to
# run a nice sanity check on your cleaning so far.


# TODO 7: Use 'select' to get the final columns needed for modeling into a new DataFrame
# called model_ready_df. These should be (in order):
#   age, bmi, bp_systolic, bp_diastolic, glucose_mgdl,
#   smoker_risk_multiplier, diabetic
model_ready_df = patients_clean  # <-- replace with patients_clean.select(...)



print("Cleaned data")
model_ready_df.show(10)
print(f"Row count after cleaning: {model_ready_df.count()} (started with {patients_raw.count()})")


# ---------------------------------------------------------------------------
# MLlib modelling -- this part is ready to go
# ---------------------------------------------------------------------------

# Define the input features that will be used to predict diabetes.
feature_cols = ["age", "bmi", "bp_systolic", "bp_diastolic", "glucose_mgdl", "smoker_risk_multiplier"]

# Create a VectorAssembler to combine all the feature columns
# into a single vector column called "features".
# Spark ML models expect input features in vector format.
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Apply the assembler to the model-ready DataFrame.
# Select only the combined feature vector and the target variable.
#
# The "diabetic" column is converted to double because Spark ML
# classification models expect the label to be numeric.
# The label column is renamed to "label", which is the standard
# column name used by Spark ML models.
assembled_df = assembler.transform(model_ready_df).select(
    "features",
    F.col("diabetic").cast("double").alias("label")
)

# Split the data into two separate datasets:
# 80% for training the logistic regression model.
# 20% for testing the model on unseen data.
#
# seed=42 ensures that the same split can be reproduced.
train_df, test_df = assembled_df.randomSplit([0.8, 0.2], seed=42)

# Create a Logistic Regression model.
# Logistic regression is used here because the target variable
# is binary: diabetic or not diabetic.
#
# featuresCol specifies the input feature vector.
# labelCol specifies the target variable.
# maxIter=20 sets the maximum number of training iterations.
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20)

# Train the logistic regression model using the training dataset.
# The model learns the relationship between the health features
# and the diabetes label.
lr_model = lr.fit(train_df)

# Use the trained model to make predictions on the test dataset.
# This produces predicted classes and prediction probabilities.
predictions = lr_model.transform(test_df)

# Display the first 10 predictions.
# features: the input feature vector.
# label: the actual diabetes outcome.
# prediction: the predicted class, usually 0.0 or 1.0.
# probability: the model's probability for each class.
#
# truncate=False prevents Spark from shortening the feature
# vectors or probability values in the displayed output.
predictions.select("features", "label", "prediction", "probability").show(10, truncate=False)

# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

# Create an evaluator to calculate the Area Under the ROC Curve (AUC).
# AUC measures how well the model distinguishes between the
# two classes across different classification thresholds.
# A higher AUC generally indicates better discrimination.
auc_evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

# Create an evaluator to calculate classification accuracy.
# Accuracy is the proportion of predictions that match
# the actual labels.
acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

# Calculate the AUC score using the test predictions.
auc = auc_evaluator.evaluate(predictions)

# Calculate the accuracy score using the test predictions.
accuracy = acc_evaluator.evaluate(predictions)

print(f"\nModel AUC: {auc:.3f}")
print(f"Model Accuracy: {accuracy:.3f}")
print("\nLearned coefficients:")

# Loop through each feature name and its corresponding
# coefficient learned by the logistic regression model.
#
# zip() pairs each feature name with its coefficient.
# The coefficient shows the direction and strength of the
# feature's relationship with the model's log-odds of diabetes.
for name, coef in zip(feature_cols, lr_model.coefficients):
    print(f"  {name:>25}: {coef:.4f}")
print(f"  {'intercept':>25}: {lr_model.intercept:.4f}")


# ---------------------------------------------------------------------------
# Bonus: Other MLlib functions worth exploring here:
# ---------------------------------------------------------------------------
# - pyspark.ml.classification.RandomForestClassifier / GBTClassifier /
#     DecisionTreeClassifier
#       Swap in for LogisticRegression above and compare AUC/accuracy.
#
# - pyspark.ml.feature.StandardScaler or MinMaxScaler
#       Scale the "features" vector before training -- often improves
#       logistic regression convergence and coefficient interpretability.
#
# - pyspark.ml.clustering.KMeans
#       Drop the "diabetic" label entirely and cluster patients by
#       age/bmi/glucose to discover natural risk groups.
#
# - pyspark.ml.regression.LinearRegression
#       Instead of classifying "diabetic" (0/1), try predicting
#       glucose_mgdl as a continuous value from the other features.
#
# ---------------------------------------------------------------------------

spark.stop()