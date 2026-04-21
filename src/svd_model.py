import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle

# Load data
print("Loading data...")
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
print(f"Train size: {len(train_df)}")
print(f"Test size: {len(test_df)}")

# Prepare data for Surprise library
reader = Reader(rating_scale=(1, 5))
train_data = Dataset.load_from_df(
    train_df[['user_id', 'item_id', 'rating']], reader
)

# Build full trainset
trainset = train_data.build_full_trainset()

# Build and train SVD model
print("\nTraining SVD model...")
svd = SVD(
    n_factors=100,
    n_epochs=20,
    lr_all=0.005,
    reg_all=0.02,
    random_state=42,
    verbose=True
)
svd.fit(trainset)
print("SVD training done!")

# Predict on test set
print("\nPredicting on test set...")
predictions = []
for _, row in test_df.iterrows():
    pred = svd.predict(row['user_id'], row['item_id'])
    predictions.append(pred.est)

test_df = test_df.copy()
test_df['svd_pred'] = predictions

# Calculate RMSE and MAE
rmse = np.sqrt(mean_squared_error(test_df['rating'], test_df['svd_pred']))
mae = mean_absolute_error(test_df['rating'], test_df['svd_pred'])
print(f"\nSVD --> RMSE: {rmse:.4f} | MAE: {mae:.4f}")

# Save results
test_df[['user_id', 'item_id', 'rating', 'svd_pred']].to_csv(
    "data/svd_predictions.csv", index=False
)
print("Saved svd_predictions.csv!")

# Save model
with open("models/svd_model.pkl", "wb") as f:
    pickle.dump(svd, f)
print("Saved svd_model.pkl!")

# Summary
print("\n--- Results Summary ---")
print(f"{'Model':<20} {'RMSE':<10} {'MAE':<10}")
print(f"{'Global Average':<20} {1.2072:<10.4f} {0.9500:<10.4f}")
print(f"{'User Average':<20} {1.2749:<10.4f} {0.8906:<10.4f}")
print(f"{'SVD':<20} {rmse:<10.4f} {mae:<10.4f}")
