import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # Ignore warnings for cleaner output

import sys
import pyspark
from pyspark.ml.recommendation import ALS  # Import ALS (Alternating Least Squares) model from Spark ML
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType, LongType  # Define data schema

from recommenders.utils.timer import Timer  # Timer utility to measure execution time
from recommenders.datasets import movielens  # Load the MovieLens dataset
from recommenders.utils.notebook_utils import is_jupyter  # Check if the code runs in Jupyter
from recommenders.datasets.spark_splitters import spark_random_split  # Function to split dataset into train/test
from recommenders.evaluation.spark_evaluation import SparkRatingEvaluation, SparkRankingEvaluation  # Metrics for model evaluation
from recommenders.utils.spark_utils import start_or_get_spark  # Start or get an existing Spark session
from recommenders.utils.notebook_utils import store_metadata  # Store metadata for evaluation

print(f"System version: {sys.version}")  # Print Python version
print("Spark version: {}".format(pyspark.__version__))  # Print Spark version

# Configuration parameters
TOP_K = 10  # Number of top recommendations to consider
RANK = 25  # Number of latent factors (dimensions) for ALS
MAX_ITER = 15  # Number of iterations for ALS training
REG_PARAM = 0.03  # Regularization parameter to prevent overfitting
MOVIELENS_DATA_SIZE = '100k'  # Choose MovieLens dataset size (100k, 1m, 10m, etc.)
POPULATION_SAMPLING_PERCENTAGE = 0.4  # Percentage of item population used for recommendations

# Column names in the dataset
COL_USER = "UserId"
COL_ITEM = "MovieId"
COL_RATING = "Rating"
COL_TIMESTAMP = "Timestamp"

# Start Spark session
spark = start_or_get_spark("ALS PySpark", memory="16g")  # Start a Spark session with 16GB of memory
spark.conf.set("spark.sql.analyzer.failAmbiguousSelfJoin", "false")  # Avoid errors due to ambiguous self-joins

# Define schema for the dataset (ensuring correct data types)
schema = StructType([
    StructField(COL_USER, IntegerType()),
    StructField(COL_ITEM, IntegerType()),
    StructField(COL_RATING, FloatType()),
    StructField(COL_TIMESTAMP, LongType()),
])

# Load MovieLens dataset into a Spark DataFrame
data = movielens.load_spark_df(spark, size=MOVIELENS_DATA_SIZE, schema=schema)
data.show()

# Split dataset into training (75%) and testing (25%) sets
train, test = spark_random_split(data, ratio=0.75, seed=123)
print("N train", train.cache().count())
print("N test", test.cache().count())

# Define ALS model parameters
header = {
    "userCol": COL_USER,
    "itemCol": COL_ITEM,
    "ratingCol": COL_RATING,
}

# Initialize ALS model with the given parameters
als = ALS(
    rank=RANK,  # Number of latent factors
    maxIter=MAX_ITER,  # Number of iterations
    implicitPrefs=True,  # Use implicit feedback (treat ratings as confidence levels)
    alpha=10,  # Confidence parameter for implicit feedback
    regParam=REG_PARAM,  # Regularization parameter to prevent overfitting
    coldStartStrategy='drop',  # Drop users/items with missing factors
    nonnegative=False,  # Allow negative factor values
    seed=42,  # seed for reproducibility
    **header  # Pass column settings
)

# Train the ALS model
with Timer() as train_time:
    model = als.fit(train)
print("Took {} seconds for training.".format(train_time.interval))

# Generate top-K recommendations
with Timer() as test_time:
    users = test.select(COL_USER).distinct()

    # Sample a percentage of items to avoid excessive computation
    items = train.select(COL_ITEM).distinct()
    items_sampled = items.sample(False, POPULATION_SAMPLING_PERCENTAGE, seed=42)

    # Create all possible user-item pairs
    user_item = users.crossJoin(items_sampled)
    # Predict ratings for these pairs
    dfs_pred = model.transform(user_item)

    print(f"Total predictions before filtering: {dfs_pred.count()}")  # Count total predictions

    # Remove previously seen (highly rated) items from recommendations
    dfs_pred_exclude_train = dfs_pred.alias("pred").join(
        train.alias("train"),
        (dfs_pred[COL_USER] == train[COL_USER]) & (dfs_pred[COL_ITEM] == train[COL_ITEM]),
        how='left'
    ).filter((F.col("train." + COL_RATING).isNull()) | (F.col("train." + COL_RATING) < 3.5)) \
        .select("pred." + COL_USER, "pred." + COL_ITEM, "pred.prediction")

    print(f"Total predictions after filtering: {dfs_pred_exclude_train.count()}")  # Count predictions after filtering

    if dfs_pred_exclude_train.count() == 0:
        raise ValueError("Filtered predictions are empty. Adjust filtering logic.")

    dfs_pred_exclude_train.cache().count()  # Cache filtered predictions
print("Took {} seconds for prediction.".format(test_time.interval))  # Print prediction time

# Evaluate ranking quality using SparkRankingEvaluation
rank_eval = SparkRankingEvaluation(
    test, dfs_pred_exclude_train, k=TOP_K,
    col_user=COL_USER, col_item=COL_ITEM, col_rating=COL_RATING,
    col_prediction="prediction", relevancy_method="top_k"
)

# numbers I got are
# MAP - Mean Average Precision
# MAP - Measures ranking quality—higher values mean relevant recommendations are ranked higher.
# MAP:    0.088367 - somewhat relevant, but there’s room for improvement.
# NDCG - Normalized Discounted Cumulative Gain
# NDCG - Evaluates how well the ranking structure matches user preferences—higher values are better.
# NDCG:   0.191767 - somewhat relevant, but there’s room for improvement.
# Precision@K:    0.174310 - 17.43% of recommended items are actually relevant.
# Recall@K:       0.100463 - The model retrieves 10.05% of all relevant items—higher recall means the model finds more relevant items.
# Precision is relatively good (~17%), meaning many recommendations are useful.
# Recall is low (~10%), suggesting that the model is missing relevant items.
# Print ranking evaluation metrics
print("Model:\tALS",
      "Top K:\t%d" % rank_eval.k,
      "MAP:\t%f" % rank_eval.map_at_k(),
      "NDCG:\t%f" % rank_eval.ndcg_at_k(),
      "Precision@K:\t%f" % rank_eval.precision_at_k(),
      "Recall@K:\t%f" % rank_eval.recall_at_k(), sep='\n')

# Evaluate rating prediction quality using SparkRatingEvaluation
prediction = model.transform(test).dropna()  # Generate predictions for test set
rating_eval = SparkRatingEvaluation(test, prediction,
                                    col_user=COL_USER, col_item=COL_ITEM,
                                    col_rating=COL_RATING, col_prediction="prediction")

# Print rating prediction evaluation metrics
# numbers I got when I ran it
# RMSE - Root Mean Squared Error
# RMSE - Measures how far the predicted ratings are from actual ratings—lower is better.
# RMSE:   2.965299 -  meaning predictions are far from actual ratings
# MAE - Mean Absolute Error
# MAE - Measures the average absolute difference between predicted and actual ratings.
# MAE:    2.752645 - meaning predictions are far from actual ratings
# Explained variance:     0.032426 - Measures how well the model captures variance in ratings—closer to 1 is better.
# R squared:      -5.956651 - Indicates how well the model fits the data—negative values mean it performs worse than a simple mean predictor.

# This poor performance is expected because ALS with implicitPrefs=True is not designed to predict exact ratings but rather to rank items.
print("Model:\tALS rating prediction",
      "RMSE:\t%f" % rating_eval.rmse(),
      "MAE:\t%f" % rating_eval.mae(),
      "Explained variance:\t%f" % rating_eval.exp_var(),
      "R squared:\t%f" % rating_eval.rsquared(), sep='\n')

# Store metadata if running in Jupyter
if is_jupyter():
    store_metadata("map", rank_eval.map_at_k())
    store_metadata("ndcg", rank_eval.ndcg_at_k())
    store_metadata("precision", rank_eval.precision_at_k())
    store_metadata("recall", rank_eval.recall_at_k())
    store_metadata("train_time", train_time.interval)
    store_metadata("test_time", test_time.interval)


# Save the trained ALS model
model_path = "als_trained_model"
model.write().overwrite().save(model_path)

# Load the trained ALS model
from pyspark.ml.recommendation import ALSModel

loaded_model = ALSModel.load(model_path)
print("Model successfully loaded!")

# Specify the user ID for whom we want recommendations
user_id = 196  # Example: User ID 196 (Change this to any valid UserId)

# Prepare the user's candidate items (items not yet rated)
user_df = spark.createDataFrame([(user_id,)], [COL_USER])  # Create a Spark DataFrame for the user
items_df = train.select(COL_ITEM).distinct()  # Get all unique items (movies)
user_items_df = user_df.crossJoin(items_df)  # Create all possible (user, item) pairs

# Generate predictions for the user
user_predictions = loaded_model.transform(user_items_df).dropna()  # Remove any null predictions

# Get the top K recommendations
top_k_recommendations = user_predictions.orderBy(F.desc("prediction")).limit(TOP_K)

# Display the recommendations
print(f"Top {TOP_K} recommended movies for User {user_id}:")
top_k_recommendations.select(COL_ITEM, "prediction").show()


# Stop the Spark session
spark.stop()
