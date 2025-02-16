import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # Ignore warnings

import sys
import pyspark
from pyspark.ml.recommendation import ALS  # Import ALS (Alternating Least Squares) from Spark ML
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType, LongType  # Define schema

from recommenders.utils.timer import Timer  # Timer utility
from recommenders.datasets import movielens  # Load MovieLens dataset
from recommenders.utils.spark_utils import start_or_get_spark  # Spark session utility
from recommenders.datasets.spark_splitters import spark_random_split  # Split dataset
from recommenders.evaluation.spark_evaluation import SparkRatingEvaluation  # Evaluation metrics

import xgboost as xgb  # Import XGBoost
import pandas as pd  # Pandas for handling data
import numpy as np  # NumPy for array operations

# Start Spark session
spark = start_or_get_spark("ALS-XGBoost", memory="16g")

# Configuration parameters
TOP_K = 10  # Number of top recommendations to consider
RANK = 25  # Number of latent factors (dimensions) for ALS
MAX_ITER = 15  # Number of iterations for ALS training
REG_PARAM = 0.03  # Regularization parameter to prevent overfitting
MOVIELENS_DATA_SIZE = '100k'  # MovieLens dataset size

COL_USER = "UserId"
COL_ITEM = "MovieId"
COL_RATING = "Rating"

# Load dataset with defined schema
schema = StructType([
    StructField(COL_USER, IntegerType()),
    StructField(COL_ITEM, IntegerType()),
    StructField(COL_RATING, FloatType()),
    StructField("Timestamp", LongType()),
])
data = movielens.load_spark_df(spark, size=MOVIELENS_DATA_SIZE, schema=schema)

# Split dataset into training (75%) and testing (25%) sets
train, test = spark_random_split(data, ratio=0.75, seed=123)

# Train ALS model with specified hyperparameters
als = ALS(
    rank=RANK,
    maxIter=MAX_ITER,
    implicitPrefs=False,  # Explicit feedback for better ratings
    regParam=REG_PARAM,
    coldStartStrategy='drop',
    userCol=COL_USER,
    itemCol=COL_ITEM,
    ratingCol=COL_RATING
)

with Timer() as train_time:
    als_model = als.fit(train)
print(f"Took {train_time.interval} seconds for ALS training.")

# Extract user and item embeddings with unique feature column names
user_factors = als_model.userFactors.withColumnRenamed("id", COL_USER) \
    .withColumnRenamed("features", "user_features")
item_factors = als_model.itemFactors.withColumnRenamed("id", COL_ITEM) \
    .withColumnRenamed("features", "item_features")

# Join ALS features with training data
train_with_factors = train.join(user_factors, on=COL_USER, how="left") \
    .join(item_factors, on=COL_ITEM, how="left")

# Convert Spark DataFrame to Pandas DataFrame for XGBoost training
train_pd = train_with_factors.toPandas()

# Function to expand list-based feature columns into multiple numerical columns
def expand_features(df, feature_col_name, prefix):
    # Replace None/NaN with zero vectors
    df[feature_col_name] = df[feature_col_name].apply(lambda x: x if x is not None else [0] * RANK)

    # Ensure all feature lists have the same length
    num_features = len(df[feature_col_name].iloc[0])
    feature_cols = [f"{prefix}_factor_{i}" for i in range(num_features)]

    # Convert list column into multiple columns
    feature_df = pd.DataFrame(df[feature_col_name].tolist(), columns=feature_cols)

    # Drop original list column and merge expanded features
    df = df.drop(columns=[feature_col_name])
    df = pd.concat([df, feature_df], axis=1)

    return df


# Expand ALS embeddings
train_pd = expand_features(train_pd, "user_features", "user")
train_pd = expand_features(train_pd, "item_features", "item")

# Drop non-feature columns and prepare training data
X_train = train_pd.drop(columns=[COL_USER, COL_ITEM, COL_RATING, "Timestamp"], errors="ignore")
y_train = train_pd[COL_RATING]

# Ensure all feature columns are of type float
X_train = X_train.astype(float)

# Train XGBoost model
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, max_depth=6)
xgb_model.fit(X_train, y_train)

print("✅ XGBoost model trained successfully!")

# Make predictions with XGBoost on test data
test_with_factors = test.join(user_factors, on=COL_USER, how="left") \
    .join(item_factors, on=COL_ITEM, how="left")
test_pd = test_with_factors.toPandas()

# Expand ALS embeddings in test data
test_pd = expand_features(test_pd, "user_features", "user")
test_pd = expand_features(test_pd, "item_features", "item")

X_test = test_pd.drop(columns=[COL_USER, COL_ITEM, COL_RATING, "Timestamp"], errors="ignore").astype(float)
test_pd["predicted_rating"] = xgb_model.predict(X_test)

# Evaluate predictions using common regression metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

rmse = mean_squared_error(test_pd[COL_RATING], test_pd["predicted_rating"]) ** 0.5  # Manually compute RMSE
mae = mean_absolute_error(test_pd[COL_RATING], test_pd["predicted_rating"])
r2 = r2_score(test_pd[COL_RATING], test_pd["predicted_rating"])

print(f"XGBoost RMSE: {rmse:.4f}")
print(f"XGBoost MAE: {mae:.4f}")
print(f"XGBoost R^2: {r2:.4f}")

# Make recommendations for a specific user
user_id = 196  # Example User ID
user_df = spark.createDataFrame([(user_id,)], [COL_USER])
items_df = train.select(COL_ITEM).distinct()
user_items_df = user_df.crossJoin(items_df)

# Add ALS factors for the user-item pairs
user_items_with_factors = user_items_df.join(user_factors, on=COL_USER, how="left") \
    .join(item_factors, on=COL_ITEM, how="left")
user_pd = user_items_with_factors.toPandas()

# Expand ALS embeddings in recommendation data
user_pd = expand_features(user_pd, "user_features", "user")
user_pd = expand_features(user_pd, "item_features", "item")

# Prepare features for prediction
X_user = user_pd.drop(columns=[COL_USER, COL_ITEM, "Timestamp"], errors="ignore").astype(float)
user_pd["predicted_rating"] = xgb_model.predict(X_user)

# Get the top-K recommendations for the user
top_k_recommendations = user_pd.sort_values(by="predicted_rating", ascending=False).head(TOP_K)

print(f"Top {TOP_K} recommendations for User {user_id}:")
print(top_k_recommendations[[COL_ITEM, "predicted_rating"]])

# Stop Spark session
spark.stop()
