import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---- Page Config ----
st.set_page_config(
    page_title="Game Rating Predictor",
    page_icon="🎮",
    layout="centered"
)

# ---- Load Models ----
@st.cache_resource
def load_svd():
    with open("models/svd_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_roberta():
    model_path = "models/roberta_finetuned"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=1
    )
    model.eval()
    return tokenizer, model

@st.cache_data
def load_weights():
    with open("models/hybrid_weights.json", "r") as f:
        return json.load(f)

@st.cache_data
def load_sample_ids():
    df = pd.read_csv("data/clean_reviews.csv")
    return df['user_id'].sample(20, random_state=42).tolist(), \
           df['item_id'].sample(20, random_state=42).tolist()

# ---- Prediction Functions ----
def predict_svd(svd, user_id, item_id):
    pred = svd.predict(user_id, item_id)
    return pred.est

def predict_roberta(tokenizer, model, review_text):
    inputs = tokenizer(
        review_text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
    pred = outputs.logits.squeeze(-1).item()
    pred = pred * 4.0 + 1.0
    return np.clip(pred, 1.0, 5.0)

def predict_hybrid(svd_pred, roberta_pred, w_roberta, w_svd):
    return np.clip(
        roberta_pred * w_roberta + svd_pred * w_svd,
        1.0, 5.0
    )

def rating_to_stars(rating):
    full = int(rating)
    return "⭐" * full

# ---- UI ----
st.title("🎮 Game Rating Predictor")
st.markdown("### Hybrid CF + LLM Rating Prediction")
st.markdown("*Amazon Video Games Dataset | IIIT Delhi*")
st.divider()

# Load models
with st.spinner("Loading models..."):
    svd = load_svd()
    weights = load_weights()
    sample_users, sample_items = load_sample_ids()

st.success("Models loaded! ✅")

# ---- Input Section ----
st.markdown("## 📝 Enter Review Details")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**User ID**")
    user_option = st.selectbox(
        "Choose a sample user or type your own",
        ["Type my own"] + sample_users,
        key="user_select"
    )
    if user_option == "Type my own":
        user_id = st.text_input("Enter User ID:", value="A1HP7NVNPFMA4N")
    else:
        user_id = user_option
    st.caption(f"Selected: `{user_id}`")

with col2:
    st.markdown("**Game ID (ASIN)**")
    item_option = st.selectbox(
        "Choose a sample game or type your own",
        ["Type my own"] + sample_items,
        key="item_select"
    )
    if item_option == "Type my own":
        item_id = st.text_input("Enter Game ID:", value="B00GMFKYJ4")
    else:
        item_id = item_option
    st.caption(f"Selected: `{item_id}`")

st.markdown("**Your Review**")
review_text = st.text_area(
    "Write your review here:",
    value="This game is absolutely amazing! The graphics are stunning and gameplay is super fun. I played for hours without getting bored!",
    height=120
)

st.divider()

# ---- Predict Button ----
if st.button("🎯 Predict Rating", type="primary", use_container_width=True):
    if not review_text.strip():
        st.error("Please write a review!")
    else:
        with st.spinner("Predicting..."):
            svd_pred = predict_svd(svd, user_id, item_id)
            tokenizer, roberta_model = load_roberta()
            roberta_pred = predict_roberta(tokenizer, roberta_model, review_text)
            hybrid_pred = predict_hybrid(
                svd_pred, roberta_pred,
                weights['w_roberta'], weights['w_svd']
            )

        st.divider()
        st.markdown("## 🏆 Prediction Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("SVD (CF Model)", f"{svd_pred:.2f} / 5.0",
                     delta=f"{svd_pred - 3:.2f} from avg")
        with col2:
            st.metric("RoBERTa (LLM)", f"{roberta_pred:.2f} / 5.0",
                     delta=f"{roberta_pred - 3:.2f} from avg")
        with col3:
            st.metric("🎯 Hybrid (Final)", f"{hybrid_pred:.2f} / 5.0",
                     delta=f"{hybrid_pred - 3:.2f} from avg")

        st.markdown("### " + rating_to_stars(round(hybrid_pred)))
        st.progress(hybrid_pred / 5.0)

        if hybrid_pred >= 4.5:
            st.success("🌟 Excellent! This user would love this game!")
        elif hybrid_pred >= 3.5:
            st.info("👍 Good! This user would enjoy this game!")
        elif hybrid_pred >= 2.5:
            st.warning("😐 Average. This user might like it.")
        else:
            st.error("👎 Poor. This game might not suit this user.")

        st.divider()

# ---- Model Performance ----
st.markdown("## 📊 Model Performance")
results_data = {
    "Model": ["Global Average", "User Average", "SVD (tuned)",
               "RoBERTa fine-tuned", "Hybrid (CF + LLM)"],
    "RMSE": [1.2072, 1.2749, 1.1601, 0.7758, 0.7710],
    "MAE": [0.9500, 0.8906, 0.8886, 0.4690, 0.4983]
}
results_df = pd.DataFrame(results_data)
st.dataframe(results_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Built by Sahil & Priyanshu | IIIT Delhi | Prof. Angshul Majumdar")
