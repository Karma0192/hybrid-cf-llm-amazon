import json
import pandas as pd
import random

# Load full dataset
print("Loading dataset...")
records = []
with open("data/Video_Games_5.json", "r") as f:
    for line in f:
        records.append(json.loads(line.strip()))

print(f"Total reviews in full dataset: {len(records)}")

# Sample 40,000 reviews
random.seed(42)
sampled = random.sample(records, 40000)
print(f"Sampled reviews: {len(sampled)}")

# Convert to DataFrame
df = pd.DataFrame(sampled)
print(f"\nColumns: {list(df.columns)}")
print(f"\nRating distribution:")
print(df['overall'].value_counts().sort_index())
print(f"\nUnique users: {df['reviewerID'].nunique()}")
print(f"Unique games: {df['asin'].nunique()}")
print(f"\nSample review:")
print(df[['reviewerID','asin','overall','reviewText']].iloc[0])

# Save sampled data
df.to_csv("data/sampled_reviews.csv", index=False)
print("\nSaved sampled_reviews.csv!")
