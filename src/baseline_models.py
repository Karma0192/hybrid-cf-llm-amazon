import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load clean data
print("Loading clean data...")
df = pd.read_csv("data/clean_reviews.csv")
print(f"Total reviews: {len(df)}")

# Split into train and test (80% train, 20% test)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Train size: {len(train_df)}")
print(f"Test size: {len(test_df)}")

# Save train and test splits for later use
train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)
print("Saved train.csv and test.csv!")

# ---- Model 1: Global Average ----
global_avg = train_df['rating'].mean()
print(f"\nGlobal Average Rating: {global_avg:.4f}")

# Predict global average for all test users
test_df = test_df.copy()
test_df['global_avg_pred'] = global_avg

# Calculate RMSE and MAE
rmse_global = np.sqrt(mean_squared_error(test_df['rating'], test_df['global_avg_pred']))
mae_global = mean_absolute_error(test_df['rating'], test_df['global_avg_pred'])
print(f"Global Average --> RMSE: {rmse_global:.4f} | MAE: {mae_global:.4f}")

# ---- Model 2: User Average ----
user_avg = train_df.groupby('user_id')['rating'].mean()

def predict_user_avg(row):
    if row['user_id'] in user_avg:
        return user_avg[row['user_id']]
    else:
        return global_avg  # fallback to global avg if user not seen

test_df['user_avg_pred'] = test_df.apply(predict_user_avg, axis=1)

rmse_user = np.sqrt(mean_squared_error(test_df['rating'], test_df['user_avg_pred']))
mae_user = mean_absolute_error(test_df['rating'], test_df['user_avg_pred'])
print(f"User Average    --> RMSE: {rmse_user:.4f} | MAE: {mae_user:.4f}")

# ---- Summary Table ----
print("\n--- Results Summary ---")
print(f"{'Model':<20} {'RMSE':<10} {'MAE':<10}")
print(f"{'Global Average':<20} {rmse_global:<10.4f} {mae_global:<10.4f}")
print(f"{'User Average':<20} {rmse_user:<10.4f} {mae_user:<10.4f}")
