"""
evaluate_ranking.py  —  NEW file, does NOT touch any existing files.

Computes NDCG@K and Hit Rate@K using leave-one-out + negative sampling.

For each user in test set:
  - Their test item is the "positive"
  - Sample 99 random items they have NOT interacted with as "negatives"
  - Rank all 100 items by predicted score
  - Compute NDCG@K and Hit Rate@K
  - Average across all users

This is the standard CF evaluation protocol when test sets are sparse.

Input files (read-only):
  data/train_features.csv
  data/test_features.csv
  data/roberta_test_preds.csv
  data/roberta_train_preds.csv
  models/svd_model.pkl
  models/hybrid_weights.json

Output (new files only):
  deliverables/ranking_metrics.json
  deliverables/ranking_metrics.txt
"""

import pandas as pd
import numpy as np
import json, os, pickle

# ───────────────────────── config ─────────────────────────
THRESHOLD    = 4.0
K_VALUES     = [5, 10]
N_NEGATIVES  = 99
RANDOM_SEED  = 42
W_ROBERTA    = 0.90
W_SVD        = 0.10
# ──────────────────────────────────────────────────────────

print("Loading data...")
train_df = pd.read_csv("data/train_features.csv")
test_df  = pd.read_csv("data/test_features.csv")

train_roberta = pd.read_csv("data/roberta_train_preds.csv").drop_duplicates(subset=["user_id","item_id"])
test_roberta  = pd.read_csv("data/roberta_test_preds.csv").drop_duplicates(subset=["user_id","item_id"])

# merge roberta predictions
train_df = train_df.merge(
    train_roberta[["user_id","item_id","roberta_pred"]],
    on=["user_id","item_id"], how="left"
).drop_duplicates(subset=["user_id","item_id"])

test_df = test_df.merge(
    test_roberta[["user_id","item_id","roberta_pred"]],
    on=["user_id","item_id"], how="left"
).drop_duplicates(subset=["user_id","item_id"])

train_df["roberta_pred"] = train_df["roberta_pred"].fillna(4.22)
test_df["roberta_pred"]  = test_df["roberta_pred"].fillna(4.22)

print(f"Train: {len(train_df)} | Test: {len(test_df)}")

# binarize — only keep relevant test items (rating >= 4) as positives
test_df["relevant"]  = (test_df["rating"] >= THRESHOLD).astype(int)
train_df["relevant"] = (train_df["rating"] >= THRESHOLD).astype(int)

global_avg   = train_df["rating"].mean()
all_items    = list(set(train_df["item_id"].tolist() + test_df["item_id"].tolist()))

print(f"Global avg : {global_avg:.4f}")
print(f"Total items: {len(all_items)}")

# per-user seen items (train + test) — negatives must not be in these
user_seen = (
    pd.concat([train_df[["user_id","item_id"]], test_df[["user_id","item_id"]]])
    .groupby("user_id")["item_id"]
    .apply(set)
    .to_dict()
)

# user average from train
user_avg_map = train_df.groupby("user_id")["rating"].mean().to_dict()

# load SVD
print("\nLoading SVD model...")
with open("models/svd_model.pkl", "rb") as f:
    svd = pickle.load(f)

# load hybrid weights
if os.path.exists("models/hybrid_weights.json"):
    with open("models/hybrid_weights.json") as f:
        w = json.load(f)
    W_ROBERTA = w["w_roberta"]
    W_SVD     = w["w_svd"]

print(f"Hybrid weights: RoBERTa={W_ROBERTA}, SVD={W_SVD}")

# ── metric functions ──
def ndcg_at_k(scores, relevance, k):
    order   = np.argsort(scores)[::-1][:k]
    rel_k   = relevance[order]
    ideal_k = np.sort(relevance)[::-1][:k]
    dcg  = sum(r / np.log2(i + 2) for i, r in enumerate(rel_k))
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal_k))
    return dcg / idcg if idcg > 0 else 0.0

def hit_at_k(scores, relevance, k):
    order = np.argsort(scores)[::-1][:k]
    return 1.0 if relevance[order].sum() > 0 else 0.0

# ── main evaluation loop ──
def evaluate_all_models(test_df):
    rng = np.random.default_rng(seed=RANDOM_SEED)

    metrics = {
        "Global Average":     {f"NDCG@{k}": [] for k in K_VALUES},
        "User Average":       {f"NDCG@{k}": [] for k in K_VALUES},
        "SVD (tuned)":        {f"NDCG@{k}": [] for k in K_VALUES},
        "RoBERTa fine-tuned": {f"NDCG@{k}": [] for k in K_VALUES},
        "Hybrid (CF + LLM)":  {f"NDCG@{k}": [] for k in K_VALUES},
    }
    for m in metrics:
        for k in K_VALUES:
            metrics[m][f"Hit Rate@{k}"] = []

    # only evaluate on relevant test items (positives)
    relevant_test = test_df[test_df["relevant"] == 1].copy()
    print(f"\nEvaluating on {len(relevant_test)} relevant test interactions...")

    for idx, (_, row) in enumerate(relevant_test.iterrows()):
        user_id  = row["user_id"]
        item_id  = row["item_id"]
        rob_pred = row["roberta_pred"]

        # sample 99 negatives not seen by this user
        seen      = user_seen.get(user_id, set())
        candidates = [it for it in all_items if it not in seen and it != item_id]

        if len(candidates) < N_NEGATIVES:
            neg_items = candidates
        else:
            neg_items = rng.choice(candidates, size=N_NEGATIVES, replace=False).tolist()

        all_eval_items = [item_id] + neg_items
        relevance = np.array([1.0] + [0.0] * len(neg_items))

        # --- Global Average ---
        scores_global = np.array([global_avg] * len(all_eval_items))
        scores_global += rng.uniform(0, 1e-9, size=len(scores_global))  # tie-break

        # --- User Average ---
        u_avg = user_avg_map.get(user_id, global_avg)
        scores_uavg = np.array([u_avg] * len(all_eval_items))
        scores_uavg += rng.uniform(0, 1e-9, size=len(scores_uavg))

        # --- SVD ---
        scores_svd = np.array([
            svd.predict(user_id, it).est for it in all_eval_items
        ])

        # --- RoBERTa: positive uses actual pred, negatives use SVD as proxy ---
        scores_rob = np.array(
            [rob_pred] + [svd.predict(user_id, it).est for it in neg_items]
        )

        # --- Hybrid ---
        scores_hybrid = np.clip(
            scores_rob * W_ROBERTA + scores_svd * W_SVD, 1.0, 5.0
        )

        score_map = {
            "Global Average":     scores_global,
            "User Average":       scores_uavg,
            "SVD (tuned)":        scores_svd,
            "RoBERTa fine-tuned": scores_rob,
            "Hybrid (CF + LLM)":  scores_hybrid,
        }

        for model_name, scores in score_map.items():
            for k in K_VALUES:
                metrics[model_name][f"NDCG@{k}"].append(ndcg_at_k(scores, relevance, k))
                metrics[model_name][f"Hit Rate@{k}"].append(hit_at_k(scores, relevance, k))

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx+1}/{len(relevant_test)}...")

    # average across users
    results = {}
    for model_name, m in metrics.items():
        results[model_name] = {
            metric: round(float(np.mean(vals)), 4)
            for metric, vals in m.items()
        }
    return results

results = evaluate_all_models(test_df)

# ── print table ──
print("\n========== NDCG & Hit Rate Results ==========")
print(f"{'Model':<25}" + "".join(f"  NDCG@{k}   HR@{k}  " for k in K_VALUES))
print("-" * 72)
for model, m in results.items():
    row = f"{model:<25}" + "".join(
        f"  {m[f'NDCG@{k}']:<10} {m[f'Hit Rate@{k}']:<8}"
        for k in K_VALUES
    )
    print(row)
print("==============================================")

# ── save ──
os.makedirs("deliverables", exist_ok=True)

with open("deliverables/ranking_metrics.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: deliverables/ranking_metrics.json")

with open("deliverables/ranking_metrics.txt", "w") as f:
    f.write("NDCG and Hit Rate Results\n")
    f.write("Protocol: Leave-one-out, 99 negative samples, binarized at rating >= 4\n\n")
    f.write(f"{'Model':<25}" + "".join(f"  NDCG@{k}   HR@{k}  " for k in K_VALUES) + "\n")
    f.write("-" * 72 + "\n")
    for model, m in results.items():
        row = f"{model:<25}" + "".join(
            f"  {m[f'NDCG@{k}']:<10} {m[f'Hit Rate@{k}']:<8}"
            for k in K_VALUES
        )
        f.write(row + "\n")
print("Saved: deliverables/ranking_metrics.txt")