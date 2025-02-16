import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import zipfile
import urllib.request

# MovieLens dataset URL & paths
MOVIELENS_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
MOVIELENS_FOLDER = "ml-100k"
MOVIELENS_FILE = os.path.join(MOVIELENS_FOLDER, "u.data")

# Download dataset if missing
if not os.path.exists(MOVIELENS_FILE):
    print("Downloading MovieLens 100K dataset...")
    urllib.request.urlretrieve(MOVIELENS_URL, "ml-100k.zip")
    print("Extracting dataset...")
    with zipfile.ZipFile("ml-100k.zip", "r") as zip_ref:
        zip_ref.extractall()
    print("Dataset ready!")

# Hyperparameters
N_FACTORS = 15  # Latent factors
N_ITERATIONS = 10  # ALS iterations
LAMBDA = 0.1  # Regularization (higher value to avoid overfitting)
TOP_K = 10  # Number of recommendations
MOVIELENS_DATA_PATH = MOVIELENS_FILE  # Path to dataset

# Load MovieLens 100K dataset
def load_movielens_data(file_path):
    """Loads MovieLens dataset and converts it to a user-item matrix."""
    df = pd.read_csv(file_path, sep="\t", names=["UserId", "MovieId", "Rating", "Timestamp"])

    num_users = df["UserId"].max()
    num_items = df["MovieId"].max()

    ratings_matrix = np.zeros((num_users, num_items))
    for _, row in df.iterrows():
        ratings_matrix[row["UserId"] - 1, row["MovieId"] - 1] = row["Rating"]

    return ratings_matrix

# ALS implementation
def als(ratings, n_factors, n_iterations, lambda_):
    """Performs ALS matrix factorization."""
    n_users, n_items = ratings.shape

    # Initialize latent factors randomly
    user_factors = np.random.rand(n_users, n_factors)
    item_factors = np.random.rand(n_items, n_factors)

    for _ in range(n_iterations):
        # Update user factors
        for i in range(n_users):
            non_zero_items = ratings[i, :] > 0
            if np.any(non_zero_items):  # Avoid empty rows
                YTY = item_factors[non_zero_items].T @ item_factors[non_zero_items]
                lambda_I = np.eye(n_factors) * lambda_
                user_factors[i] = np.linalg.solve(YTY + lambda_I,
                                                  ratings[i, non_zero_items] @ item_factors[non_zero_items])

        # Update item factors
        for j in range(n_items):
            non_zero_users = ratings[:, j] > 0
            if np.any(non_zero_users):  # Avoid empty columns
                XTX = user_factors[non_zero_users].T @ user_factors[non_zero_users]
                lambda_I = np.eye(n_factors) * lambda_
                item_factors[j] = np.linalg.solve(XTX + lambda_I,
                                                  ratings[non_zero_users, j] @ user_factors[non_zero_users])

    return user_factors, item_factors

# Load dataset
ratings_matrix = load_movielens_data(MOVIELENS_DATA_PATH)

# Run ALS
user_factors, item_factors = als(ratings_matrix, N_FACTORS, N_ITERATIONS, LAMBDA)

# Compute predicted ratings
predicted_ratings = np.dot(user_factors, item_factors.T)

# 🔹 Apply rating scaling to avoid bias
mean_pred = np.mean(predicted_ratings)
std_pred = np.std(predicted_ratings)
predicted_ratings = 3 + 2 * ((predicted_ratings - mean_pred) / std_pred)  # Scale to 1-5 range
predicted_ratings = np.clip(predicted_ratings, 1, 5)  # Ensure within valid range

# Extract non-zero ratings for evaluation
mask = ratings_matrix > 0
actual_ratings = ratings_matrix[mask]
predicted_ratings_filtered = predicted_ratings[mask]

# Compute evaluation metrics
rmse = mean_squared_error(actual_ratings, predicted_ratings_filtered) ** 0.5
mae = mean_absolute_error(actual_ratings, predicted_ratings_filtered)
r2 = r2_score(actual_ratings, predicted_ratings_filtered)

print(f"🔹 ALS RMSE: {rmse:.4f}")
print(f"🔹 ALS MAE: {mae:.4f}")
print(f"🔹 ALS R²: {r2:.4f}")

# Generate recommendations for a specific user
def recommend_movies(user_id, predicted_ratings, top_k):
    """Returns top-K recommended movies for a given user."""
    user_ratings = predicted_ratings[user_id - 1]  # Adjust for 0-based index
    top_movies = np.argsort(user_ratings)[::-1][:top_k]  # Sort by highest rating

    return pd.DataFrame({"MovieId": top_movies + 1, "PredictedRating": user_ratings[top_movies]})

# Recommend movies for User 196
user_id = 196
top_k_recommendations = recommend_movies(user_id, predicted_ratings, TOP_K)

print(f"\n🔹 Top {TOP_K} recommendations for User {user_id}:")
print(top_k_recommendations)

# Plot the original and predicted ratings distributions
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(ratings_matrix, aspect='auto', cmap="viridis")
plt.title("Original Ratings")
plt.xlabel("Movies")
plt.ylabel("Users")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(predicted_ratings, aspect='auto', cmap="viridis")
plt.title("Predicted Ratings (After Scaling)")
plt.xlabel("Movies")
plt.ylabel("Users")
plt.colorbar()

plt.show()

# Plot rating distributions before and after scaling
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(predicted_ratings_filtered, bins=20, color='blue', alpha=0.7)
plt.title("Predicted Ratings Distribution")
plt.xlabel("Rating")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(actual_ratings, bins=20, color='red', alpha=0.7)
plt.title("Actual Ratings Distribution")
plt.xlabel("Rating")
plt.ylabel("Frequency")

plt.show()
