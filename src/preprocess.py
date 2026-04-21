import pandas as pd
import json
import random

print("Loading full dataset...")
records = []
with open("data/Video_Games_5.json", "r") as f:
    for line in f:
        records.append(json.loads(line.strip()))

print(f"Total records: {len(records)}")

# Sample 40K
random.seed(42)
sampled = random.sample(records, 40000)
df = pd.DataFrame(sampled)

print(f"\nBefore cleaning: {df.shape}")

# Keep only columns we need
df = df[['reviewerID', 'asin', 'overall', 'reviewText', 'summary']]

# Rename columns to simpler names
df.columns = ['user_id', 'item_id', 'rating', 'review_text', 'summary']

# Drop rows where rating is missing
df = df.dropna(subset=['rating'])

# Drop rows where review text is missing
df = df.dropna(subset=['review_text'])

# Make sure rating is a number
df['rating'] = df['rating'].astype(float)

# Remove reviews where text is empty string
df = df[df['review_text'].str.strip() != '']

# Combine review_text and summary into one column for LLM
df['full_text'] = df['summary'].fillna('') + ' ' + df['review_text']
df['full_text'] = df['full_text'].str.strip()

print(f"After cleaning: {df.shape}")
print(f"\nRating distribution:")
print(df['rating'].value_counts().sort_index())
print(f"\nUnique users: {df['user_id'].nunique()}")
print(f"Unique items: {df['item_id'].nunique()}")
print(f"\nSample row:")
print(df.iloc[0])

# Save clean data
df.to_csv("data/clean_reviews.csv", index=False)
print("\nSaved clean_reviews.csv!")
