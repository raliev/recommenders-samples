# Recommendation System Samples

## ALS + XGBoost Movie 

This repository contains a hybrid recommendation system using **Alternating Least Squares (ALS)** and **XGBoost**.  

It trains ALS to generate latent factors and then uses XGBoost to improve rating predictions.

## Features
- Uses **MovieLens 100K dataset** for training.
- **Spark ALS** to extract user and item embeddings.
- **XGBoost** to refine rating predictions.
- Generates **personalized movie recommendations**.

---

## Setup & Installation

### How to run

```sh
git clone https://github.com/raliev/recommenders-samples.git
cd recommenders-samples
pip install -r requirements.txt
cd samples
python als.py
python als-and-xgboost.py
