import streamlit as st
import joblib
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- Page Config ---
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🧠",
    layout="centered"
)

# --- Download NLTK data ---
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# --- Load Models ---
@st.cache_resource
def load_models():
    model = joblib.load('model_lr.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    emotion_labels = joblib.load('emotion_labels.pkl')
    # Reverse the dict: {0: 'sadness', 1: 'anger', ...}
    label_map = {v: k for k, v in emotion_labels.items()}
    return model, vectorizer, label_map

model, vectorizer, label_map = load_models()

# --- Emotion Config ---
EMOTION_CONFIG = {
    'sadness':  {'emoji': '😢', 'color': '#5B8CFF'},
    'anger':    {'emoji': '😡', 'color': '#FF5B5B'},
    'love':     {'emoji': '❤️',  'color': '#FF80AB'},
    'surprise': {'emoji': '😲', 'color': '#FFD700'},
    'fear':     {'emoji': '😨', 'color': '#B39DDB'},
    'joy':      {'emoji': '😄', 'color': '#69F0AE'},
}

# --- Text Preprocessing (same as training) ---
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join([c for c in text if ord(c) < 128])
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

# --- Predict ---
def predict_emotion(text):
    cleaned = preprocess(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]
    emotion = label_map[prediction]
    return emotion, probabilities

# --- UI Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600&family=Inter:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main {
        background-color: #0f0f0f;
    }
    .title {
        font-family: 'Playfair Display', serif;
        font-size: 2.6rem;
        font-weight: 600;
        color: #f5f0e8;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }
    .result-card {
        background: #1a1a1a;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
        border: 1px solid #2a2a2a;
    }
    .emotion-emoji {
        font-size: 4rem;
    }
    .emotion-label {
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        font-weight: 600;
        margin-top: 0.5rem;
        text-transform: capitalize;
    }
    .confidence-text {
        color: #888;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }
    .model-info {
        background: #1a1a1a;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-top: 2rem;
        border: 1px solid #2a2a2a;
        font-size: 0.85rem;
        color: #888;
    }
    .model-info span {
        color: #69F0AE;
        font-weight: 500;
    }
    .stTextArea textarea {
        background-color: #1a1a1a !important;
        color: #f5f0e8 !important;
        border: 1px solid #333 !important;
        border-radius: 12px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
    }
    .stButton > button {
        background: #f5f0e8;
        color: #0f0f0f;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.6rem 2rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover {
        opacity: 0.85;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="title">🧠 Emotion Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Type something and the model will detect the emotion behind it.</div>', unsafe_allow_html=True)

# --- Input ---
user_input = st.text_area("", placeholder="e.g. I feel so hopeless today...", height=130, label_visibility="collapsed")

predict_btn = st.button("Detect Emotion")

# --- Prediction ---
if predict_btn:
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        emotion, probs = predict_emotion(user_input)
        config = EMOTION_CONFIG.get(emotion, {'emoji': '🤔', 'color': '#ffffff'})

        st.markdown(f"""
        <div class="result-card">
            <div class="emotion-emoji">{config['emoji']}</div>
            <div class="emotion-label" style="color: {config['color']};">{emotion}</div>
            <div class="confidence-text">Confidence: {max(probs)*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # Probability bar chart for all emotions
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Confidence across all emotions:**")
        
        prob_dict = {label_map[i]: float(probs[i]) for i in range(len(probs))}
        # Sort by probability
        prob_dict = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
        
        for emo, prob in prob_dict.items():
            cfg = EMOTION_CONFIG.get(emo, {'emoji': '🤔', 'color': '#ffffff'})
            st.markdown(f"{cfg['emoji']} **{emo.capitalize()}**")
            st.progress(prob)

# --- Model Info ---
st.markdown("""
<div class="model-info">
    🤖 <b>Model:</b> Logistic Regression + TF-IDF &nbsp;|&nbsp;
    <b>Accuracy:</b> <span>86.15%</span> &nbsp;|&nbsp;
    <b>Classes:</b> 6 emotions &nbsp;|&nbsp;
    <b>Dataset:</b> Emotions (train.txt)
</div>
""", unsafe_allow_html=True)
