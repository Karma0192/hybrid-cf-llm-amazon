import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("Loading data...")
test_features = pd.read_csv("data/test_features.csv")
train_features = pd.read_csv("data/train_features.csv")
test_roberta = pd.read_csv("data/roberta_test_preds.csv")
train_roberta = pd.read_csv("data/roberta_train_preds.csv")

# Merge and deduplicate
train_roberta = train_roberta.drop_duplicates(subset=['user_id','item_id'])
test_roberta = test_roberta.drop_duplicates(subset=['user_id','item_id'])

train_df = train_features.merge(
    train_roberta[['user_id','item_id','roberta_pred']],
    on=['user_id','item_id'], how='left'
).drop_duplicates(subset=['user_id','item_id'])

test_df = test_features.merge(
    test_roberta[['user_id','item_id','roberta_pred']],
    on=['user_id','item_id'], how='left'
).drop_duplicates(subset=['user_id','item_id'])

train_df['roberta_pred'] = train_df['roberta_pred'].fillna(4.22)
test_df['roberta_pred'] = test_df['roberta_pred'].fillna(4.22)
print(f"Train: {len(train_df)} | Test: {len(test_df)}")

# Load SVD
print("\nLoading SVD model...")
with open("models/svd_model.pkl", "rb") as f:
    svd = pickle.load(f)

train_df['svd_pred'] = [svd.predict(r['user_id'], r['item_id']).est
                         for _, r in train_df.iterrows()]
test_df['svd_pred'] = [svd.predict(r['user_id'], r['item_id']).est
                        for _, r in test_df.iterrows()]
print("SVD done!")

# Optimal weights found: RoBERTa=0.90, SVD=0.10
W_ROBERTA = 0.90
W_SVD = 0.10

print(f"\nApplying optimal weights: RoBERTa={W_ROBERTA}, SVD={W_SVD}")
y_test = test_df['rating'].values

# Final hybrid predictions
hybrid_preds = np.clip(
    test_df['roberta_pred'].values * W_ROBERTA +
    test_df['svd_pred'].values * W_SVD,
    1.0, 5.0
)

# Individual scores
rmse_roberta = np.sqrt(mean_squared_error(y_test,
    np.clip(test_df['roberta_pred'].values, 1.0, 5.0)))
mae_roberta = mean_absolute_error(y_test,
    np.clip(test_df['roberta_pred'].values, 1.0, 5.0))

rmse_svd = np.sqrt(mean_squared_error(y_test,
    np.clip(test_df['svd_pred'].values, 1.0, 5.0)))
mae_svd = mean_absolute_error(y_test,
    np.clip(test_df['svd_pred'].values, 1.0, 5.0))

rmse_hybrid = np.sqrt(mean_squared_error(y_test, hybrid_preds))
mae_hybrid = mean_absolute_error(y_test, hybrid_preds)

# Save predictions
test_df['hybrid_pred'] = hybrid_preds
test_df[['user_id','item_id','rating','svd_pred',
         'roberta_pred','hybrid_pred']].to_csv(
    "data/hybrid_predictions.csv", index=False)
print("Saved hybrid_predictions.csv!")

# Save weights
import json
weights = {"w_roberta": W_ROBERTA, "w_svd": W_SVD}
with open("models/hybrid_weights.json", "w") as f:
    json.dump(weights, f)
print("Saved hybrid_weights.json!")

print("\n========================================")
print("        FINAL RESULTS SUMMARY")
print("========================================")
print(f"{'Model':<25} {'RMSE':<10} {'MAE':<10}")
print(f"{'-'*45}")
print(f"{'Global Average':<25} {1.2072:<10.4f} {0.9500:<10.4f}")
print(f"{'User Average':<25} {1.2749:<10.4f} {0.8906:<10.4f}")
print(f"{'SVD (tuned)':<25} {rmse_svd:<10.4f} {mae_svd:<10.4f}")
print(f"{'RoBERTa fine-tuned':<25} {rmse_roberta:<10.4f} {mae_roberta:<10.4f}")
print(f"{'Hybrid (CF + LLM)':<25} {rmse_hybrid:<10.4f} {mae_hybrid:<10.4f}")
print(f"{'-'*45}")
print(f"Improvement over RoBERTa: {rmse_roberta - rmse_hybrid:.4f} RMSE")
print(f"Improvement over SVD:     {rmse_svd - rmse_hybrid:.4f} RMSE")
print(f"Improvement over Baseline:{1.2072 - rmse_hybrid:.4f} RMSE")
print("========================================")
