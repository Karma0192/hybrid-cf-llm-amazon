import pandas as pd
import numpy as np
import json

print("Loading clean data...")
df = pd.read_csv("data/clean_reviews.csv")
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
print(f"Total: {len(df)} | Train: {len(train_df)} | Test: {len(test_df)}")

# ---- Feature 1: Review text length ----
df['review_length'] = df['review_text'].apply(lambda x: len(str(x).split()))
train_df['review_length'] = train_df['review_text'].apply(lambda x: len(str(x).split()))
test_df['review_length'] = test_df['review_text'].apply(lambda x: len(str(x).split()))

# ---- Feature 2: User average rating (from train only) ----
user_avg = train_df.groupby('user_id')['rating'].mean()
global_avg = train_df['rating'].mean()

train_df['user_avg_rating'] = train_df['user_id'].map(user_avg).fillna(global_avg)
test_df['user_avg_rating'] = test_df['user_id'].map(user_avg).fillna(global_avg)

# ---- Feature 3: Item average rating (from train only) ----
item_avg = train_df.groupby('item_id')['rating'].mean()

train_df['item_avg_rating'] = train_df['item_id'].map(item_avg).fillna(global_avg)
test_df['item_avg_rating'] = test_df['item_id'].map(item_avg).fillna(global_avg)

# ---- Feature 4: User review count ----
user_count = train_df.groupby('user_id')['rating'].count()
train_df['user_review_count'] = train_df['user_id'].map(user_count).fillna(1)
test_df['user_review_count'] = test_df['user_id'].map(user_count).fillna(1)

# ---- Feature 5: Item review count ----
item_count = train_df.groupby('item_id')['rating'].count()
train_df['item_review_count'] = train_df['item_id'].map(item_count).fillna(1)
test_df['item_review_count'] = test_df['item_id'].map(item_count).fillna(1)

print("\nFeatures created!")
print(f"Train columns: {list(train_df.columns)}")

# Save enriched train and test
train_df.to_csv("data/train_features.csv", index=False)
test_df.to_csv("data/test_features.csv", index=False)
print("Saved train_features.csv and test_features.csv!")

# ---- Create JSONL for LLM fine-tuning ----
print("\nCreating JSONL for LLM fine-tuning...")

# Use training data only for fine-tuning
# Take 10000 samples to keep Colab training fast
llm_df = train_df.sample(n=10000, random_state=42)

with open("data/llm_finetune.jsonl", "w") as f:
    for _, row in llm_df.iterrows():
        text = f"Rating: {int(row['rating'])}. Review: {str(row['review_text'])[:300]}"
        json.dump({"text": text}, f)
        f.write("\n")

print(f"Saved llm_finetune.jsonl with 10000 examples!")
print("\nSample JSONL entry:")
sample = f"Rating: 5. Review: {str(llm_df.iloc[0]['review_text'])[:200]}"
print(sample)

# ---- Summary of features ----
print("\n--- Feature Summary ---")
print(f"review_length    - avg: {train_df['review_length'].mean():.1f} words")
print(f"user_avg_rating  - avg: {train_df['user_avg_rating'].mean():.2f}")
print(f"item_avg_rating  - avg: {train_df['item_avg_rating'].mean():.2f}")
print(f"user_review_count- avg: {train_df['user_review_count'].mean():.1f}")
print(f"item_review_count- avg: {train_df['item_review_count'].mean():.1f}")
