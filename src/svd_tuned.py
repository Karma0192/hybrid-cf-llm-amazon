import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle

print("Loading data...")
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

reader = Reader(rating_scale=(1, 5))
train_data = Dataset.load_from_df(
    train_df[['user_id', 'item_id', 'rating']], reader
)
trainset = train_data.build_full_trainset()

print("\nTraining Tuned SVD model...")
svd = SVD(
    n_factors=200,
    n_epochs=50,
    lr_all=0.005,
    reg_all=0.1,
    random_state=42,
    verbose=True
)
svd.fit(trainset)
print("Done!")

predictions = []
for _, row in test_df.iterrows():
    pred = svd.predict(row['user_id'], row['item_id'])
    predictions.append(pred.est)

test_df = test_df.copy()
test_df['svd_pred'] = predictions

rmse = np.sqrt(mean_squared_error(test_df['rating'], test_df['svd_pred']))
mae = mean_absolute_error(test_df['rating'], test_df['svd_pred'])
print(f"\nTuned SVD --> RMSE: {rmse:.4f} | MAE: {mae:.4f}")

test_df[['user_id', 'item_id', 'rating', 'svd_pred']].to_csv(
    "data/svd_predictions.csv", index=False
)
with open("models/svd_model.pkl", "wb") as f:
    pickle.dump(svd, f)
print("Saved!")

print("\n--- Results Summary ---")
print(f"{'Model':<20} {'RMSE':<10} {'MAE':<10}")
print(f"{'Global Average':<20} {1.2072:<10.4f} {0.9500:<10.4f}")
print(f"{'User Average':<20} {1.2749:<10.4f} {0.8906:<10.4f}")
print(f"{'SVD (basic)':<20} {1.1657:<10.4f} {0.8970:<10.4f}")
print(f"{'SVD (tuned)':<20} {rmse:<10.4f} {mae:<10.4f}")
