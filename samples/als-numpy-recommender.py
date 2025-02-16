import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import zipfile
import urllib.request

MOVIELENS_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
MOVIELENS_FOLDER = "ml-100k"
MOVIELENS_FILE = os.path.join(MOVIELENS_FOLDER, "u.data")

# Check if dataset exists, otherwise download and extract
if not os.path.exists(MOVIELENS_FILE):
    print("Downloading MovieLens 100K dataset...")
    urllib.request.urlretrieve(MOVIELENS_URL, "ml-100k.zip")

    print("Extracting dataset...")
    with zipfile.ZipFile("ml-100k.zip", "r") as zip_ref:
        zip_ref.extractall()

    print("Dataset ready!")

# Parameters matching Spark-based ALS
N_FACTORS = 25  # Number of latent factors
N_ITERATIONS = 15  # ALS iterations
LAMBDA = 0.03  # Regularization parameter
TOP_K = 10  # Number of recommendations
MOVIELENS_DATA_PATH = "ml-100k/u.data"  # Path to MovieLens dataset

# Load MovieLens 100K dataset
def load_movielens_data(file_path):
    """Loads MovieLens 100K dataset and converts it to a user-item matrix."""
    df = pd.read_csv(file_path, sep="\t", names=["UserId", "MovieId", "Rating", "Timestamp"])

    # Create user-item ratings matrix
    num_users = df["UserId"].max()
    num_items = df["MovieId"].max()

    ratings_matrix = np.zeros((num_users, num_items))
    for _, row in df.iterrows():
        ratings_matrix[row["UserId"] - 1, row["MovieId"] - 1] = row["Rating"]

    return ratings_matrix

# ALS implementation in NumPy
def als(ratings, n_factors, n_iterations, lambda_):
    """Performs ALS factorization."""
    n_users, n_items = ratings.shape

    # Initialize user and item latent factors
    user_factors = np.random.rand(n_users, n_factors)
    item_factors = np.random.rand(n_items, n_factors)

    for iteration in range(n_iterations):
        # Update user factors
        for i in range(n_users):
            non_zero_items = ratings[i, :] > 0
            if np.any(non_zero_items):  # Avoid empty rows
                YTY = item_factors[non_zero_items].T.dot(item_factors[non_zero_items])
                lambda_I = np.eye(n_factors) * lambda_
                user_factors[i] = np.linalg.solve(YTY + lambda_I,
                                                  ratings[i, non_zero_items].dot(item_factors[non_zero_items]))

        # Update item factors
        for j in range(n_items):
            non_zero_users = ratings[:, j] > 0
            if np.any(non_zero_users):  # Avoid empty columns
                XTX = user_factors[non_zero_users].T.dot(user_factors[non_zero_users])
                lambda_I = np.eye(n_factors) * lambda_
                item_factors[j] = np.linalg.solve(XTX + lambda_I,
                                                  ratings[non_zero_users, j].dot(user_factors[non_zero_users]))

    return user_factors, item_factors

# Load dataset
ratings_matrix = load_movielens_data(MOVIELENS_DATA_PATH)

# Run ALS
user_factors, item_factors = als(ratings_matrix, N_FACTORS, N_ITERATIONS, LAMBDA)

# Generate predicted ratings
predicted_ratings = np.dot(user_factors, item_factors.T)

# Extract non-zero ratings for evaluation
mask = ratings_matrix > 0
actual_ratings = ratings_matrix[mask]
predicted_ratings_filtered = predicted_ratings[mask]

# Compute evaluation metrics
rmse = mean_squared_error(actual_ratings, predicted_ratings_filtered) ** 0.5
mae = mean_absolute_error(actual_ratings, predicted_ratings_filtered)
r2 = r2_score(actual_ratings, predicted_ratings_filtered)

print(f"ALS RMSE: {rmse:.4f}")
print(f"ALS MAE: {mae:.4f}")
print(f"ALS R²: {r2:.4f}")

# Generate recommendations for a specific user
def recommend_movies(user_id, predicted_ratings, top_k):
    """Generates top-K recommendations for a given user."""
    user_ratings = predicted_ratings[user_id - 1]  # Adjust for 0-based index
    top_movies = np.argsort(user_ratings)[::-1][:top_k]  # Sort by highest rating

    recommendations = pd.DataFrame({
        "MovieId": top_movies + 1,  # Convert back to 1-based indexing
        "PredictedRating": user_ratings[top_movies]
    })

    return recommendations

# Example: Recommend movies for User 196
user_id = 196
top_k_recommendations = recommend_movies(user_id, predicted_ratings, TOP_K)

print(f"\nTop {TOP_K} recommendations for User {user_id}:")
print(top_k_recommendations)

# Plot the original and predicted ratings matrices
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(ratings_matrix, aspect='auto', cmap="viridis")
plt.title("Original Ratings")
plt.xlabel("Movies")
plt.ylabel("Users")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(predicted_ratings, aspect='auto', cmap="viridis")
plt.title("Predicted Ratings")
plt.xlabel("Movies")
plt.ylabel("Users")
plt.colorbar()

plt.show()
