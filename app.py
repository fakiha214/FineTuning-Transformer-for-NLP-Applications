import streamlit as st
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import gdown
import os
import zipfile
from pathlib import Path

st.set_page_config(
    page_title="BERT Sentiment Classification",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("BERT Sentiment Classification")
st.markdown("Analyze customer feedback sentiment using fine-tuned BERT model")

GOOGLE_DRIVE_FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID_HERE"
MODEL_PATH = "bert_sentiment_model"
MODEL_ZIP = "bert_sentiment_model.zip"

@st.cache_resource
def download_model_from_drive():
    """Download model from Google Drive if not already present"""
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model from Google Drive...")
        try:
            gdown.download(
                f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}",
                MODEL_ZIP,
                quiet=False
            )

            if os.path.exists(MODEL_ZIP):
                st.info("Extracting model...")
                with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
                    zip_ref.extractall()
                os.remove(MODEL_ZIP)
                st.success("Model downloaded and extracted successfully!")
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            return False
    return True

@st.cache_resource
def load_model():
    """Load BERT model and tokenizer"""
    if not download_model_from_drive():
        st.error("Could not load model. Please check the Google Drive file ID.")
        return None, None

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def predict_sentiment(text, model, tokenizer, device):
    """Predict sentiment for given text"""
    try:
        encoding = tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            pred_label = torch.argmax(logits, dim=1).item()

        sentiment_labels = ['Negative', 'Neutral', 'Positive']
        sentiment = sentiment_labels[pred_label]
        confidence = probabilities[0][pred_label].item()

        return sentiment, confidence, probabilities[0].cpu().numpy()
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None

st.sidebar.header("Configuration")
st.sidebar.markdown("### Setup Instructions")
st.sidebar.markdown("""
1. Train the model using task1_bert_sentiment_classification.ipynb in Google Colab
2. Download the trained model folder (bert_sentiment_model.zip)
3. Upload to Google Drive
4. Replace YOUR_GOOGLE_DRIVE_FILE_ID_HERE with your file ID
5. Get file ID: Right-click → Get link → Extract from URL
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Information")
st.sidebar.markdown("""
- **Model**: BERT Base (bert-base-uncased)
- **Task**: Sentiment Classification
- **Classes**: Positive, Negative, Neutral
- **Device**: GPU (if available) / CPU
""")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Input Text")
    user_input = st.text_area(
        "Enter customer feedback:",
        placeholder="e.g., I absolutely love this product! It's amazing!",
        height=120
    )

with col2:
    st.subheader("Examples")
    examples = {
        "Positive": "I absolutely love this product! It's amazing!",
        "Negative": "This is the worst experience I've ever had.",
        "Neutral": "It's okay, nothing special."
    }

    for label, text in examples.items():
        if st.button(f"Try {label}", key=label):
            user_input = text

if user_input:
    if st.button("Analyze Sentiment", key="analyze"):
        with st.spinner("Loading model and analyzing..."):
            result = load_model()

            if result[0] is not None:
                model, tokenizer, device = result
                sentiment, confidence, probabilities = predict_sentiment(
                    user_input, model, tokenizer, device
                )

                if sentiment:
                    st.success(f"Sentiment: **{sentiment}** (Confidence: {confidence:.2%})")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Negative", f"{probabilities[0]:.2%}")
                    with col2:
                        st.metric("Neutral", f"{probabilities[1]:.2%}")
                    with col3:
                        st.metric("Positive", f"{probabilities[2]:.2%}")

                    st.markdown("---")

                    st.subheader("Confidence Breakdown")
                    chart_data = pd.DataFrame({
                        'Sentiment': ['Negative', 'Neutral', 'Positive'],
                        'Confidence': probabilities
                    })
                    st.bar_chart(chart_data.set_index('Sentiment'))

st.markdown("---")

with st.expander("About this App"):
    st.markdown("""
    ### Task 1: BERT Sentiment Classification

    This application uses a fine-tuned BERT model to classify customer feedback into three sentiment categories:
    - **Positive**: Favorable customer feedback
    - **Negative**: Unfavorable customer feedback
    - **Neutral**: Neutral or mixed feedback

    ### Model Details
    - **Architecture**: BERT Base (Encoder-only)
    - **Training Data**: Customer feedback dataset with 293,855+ samples
    - **Evaluation Metrics**: Accuracy, F1-Score, Confusion Matrix

    ### How to Deploy
    1. Train the model in Google Colab using the provided notebook
    2. Download the trained model folder
    3. Upload to Google Drive
    4. Update the GOOGLE_DRIVE_FILE_ID in this script
    5. Deploy using Streamlit Cloud or Docker

    ### Technologies Used
    - PyTorch
    - Transformers (Hugging Face)
    - Streamlit
    - Google Drive API (gdown)
    """)

st.markdown("---")
st.caption("BERT Sentiment Classification | NLP Assignment A3 | Task 1")
