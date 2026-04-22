# 🎮 Hybrid CF + LLM Rating Prediction
### Amazon Video Games Dataset | IIIT Delhi

Built by **Sahil & Priyanshu** | Course: Collaborative Filtering | Prof. Angshul Majumdar

---

## 📌 Project Overview

This project builds a **Hybrid Recommendation System** that combines:
- **Collaborative Filtering (SVD)** — learns from user-item rating patterns
- **Fine-tuned LLM (RoBERTa)** — understands review text sentiment
- **Hybrid Model** — combines both for best predictions

**Task:** Predict user ratings (1-5 stars) for Amazon Video Games
**Evaluation:** RMSE and MAE

---

## 🏆 Results

| Model | RMSE | MAE |
|---|---|---|
| Global Average (Baseline) | 1.2072 | 0.9500 |
| User Average (Baseline) | 1.2749 | 0.8906 |
| SVD (Collaborative Filtering) | 1.1601 | 0.8886 |
| RoBERTa Fine-tuned (LLM) | 0.7758 | 0.4690 |
| **Hybrid (CF + LLM)** | **0.7710** | **0.4983** |

> 📉 **36% improvement** over baseline using Hybrid approach!

---

## 📁 Project Structure

```
hybrid-cf-llm-amazon/
│
├── data/                          # Dataset files (not tracked by git)
│   ├── Video_Games_5.json         # Raw Amazon dataset
│   ├── clean_reviews.csv          # Cleaned 40K sample
│   ├── train.csv                  # Training split (80%)
│   ├── test.csv                   # Test split (20%)
│   ├── train_features.csv         # Engineered features
│   ├── test_features.csv          # Engineered features
│   ├── llm_finetune.jsonl         # JSONL for RoBERTa fine-tuning
│   ├── roberta_train_preds.csv    # RoBERTa predictions on train
│   └── roberta_test_preds.csv     # RoBERTa predictions on test
│
├── models/                        # Saved models (not tracked by git)
│   ├── svd_model.pkl              # Trained SVD model
│   ├── hybrid_weights.json        # Optimal hybrid weights
│   ├── scaler.pkl                 # Feature scaler
│   └── roberta_finetuned/         # Fine-tuned RoBERTa model
│
├── src/                           # Source code
│   ├── explore_data.py            # Data exploration
│   ├── preprocess.py              # Data preprocessing
│   ├── baseline_models.py         # Global avg + User avg
│   ├── svd_model.py               # Basic SVD model
│   ├── svd_tuned.py               # Tuned SVD model
│   ├── feature_engineering.py     # Feature engineering + JSONL
│   └── hybrid_model.py            # Final hybrid model
│
├── streamlit_app/
│   └── app.py                     # Streamlit demo UI
│
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## ⚙️ Setup Instructions

### Prerequisites
- GitHub Education Pack (for Codespaces)
- Google Account (for Colab)
- Python 3.12+

---

### Step 1: Clone and open in Codespaces
```bash
# Go to GitHub repo and click Code → Codespaces → Create codespace
```

### Step 2: Activate virtual environment
```bash
source .venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download dataset
```bash
cd data
wget --no-check-certificate https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Video_Games_5.json.gz
gunzip Video_Games_5.json.gz
cd ..
```

### Step 5: Run pipeline in order
```bash
# 1. Explore data
python src/explore_data.py

# 2. Preprocess
python src/preprocess.py

# 3. Baseline models
python src/baseline_models.py

# 4. SVD model
python src/svd_tuned.py

# 5. Feature engineering
python src/feature_engineering.py

# 6. Hybrid model (after RoBERTa fine-tuning)
python src/hybrid_model.py
```

---

### Step 6: Fine-tune RoBERTa on Google Colab

1. Open [Google Colab](https://colab.research.google.com)
2. Set runtime to **T4 GPU**
3. Run the following install command:
```python
!pip uninstall -y bitsandbytes triton
!pip install transformers==4.44.0 peft==0.12.0 trl==0.9.6 accelerate==0.33.0 datasets==2.21.0
```
4. Load JSONL from GitHub:
```python
url = "https://raw.githubusercontent.com/Karma0192/hybrid-cf-llm-amazon/main/data/llm_finetune.jsonl"
```
5. Fine-tune **distilroberta-base** for 3 epochs
6. Download fine-tuned model and upload to `models/roberta_finetuned/`

---

### Step 7: Run Streamlit demo
```bash
streamlit run streamlit_app/app.py
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Development | GitHub Codespaces |
| Language | Python 3.12 |
| CF Model | SVD via scikit-surprise |
| LLM | distilroberta-base (HuggingFace) |
| Fine-tuning | Google Colab (T4 GPU) |
| Hybrid | Weighted Ensemble (90% RoBERTa + 10% SVD) |
| UI | Streamlit |

---

## 📊 Dataset

- **Source:** Amazon Video Games 5-core
- **Full size:** 497,577 reviews
- **Used:** 40,000 reviews (hardware constraint)
- **Split:** 80% train / 20% test
- **Features:** user_id, item_id, rating, review_text

> ⚠️ Due to hardware constraints (8GB RAM, GTX 1650 4GB VRAM),
> we sampled 40,000 reviews from the full dataset using
> random sampling with seed=42 for reproducibility.

---

## 📦 Requirements

```
numpy==1.26.4
pandas
scikit-learn
scikit-surprise
matplotlib
seaborn
jupyter
streamlit
torch
transformers
```

---

## 👥 Team

| Name | Role |
|---|---|
| Sahil (Karma0192) | Lead Developer |
| Priyanshu | Co-Developer |

**Professor:** Prof. Angshul Majumdar
**College:** IIIT Delhi
**Course:** Collaborative Filtering
