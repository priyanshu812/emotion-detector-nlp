import streamlit as st
import joblib
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# load models
@st.cache_resource
def load_models():
    labels = joblib.load('emotion_labels.pkl')
    vector = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('model_lr.pkl')
    return labels,vector,model

#Preprocessing
def preprocessing(text):
    text = text.lower()
    text = text.translate(str.maketrans('','',string.punctuation))
    text= ''.join([c for c in text if ord(c)<128])
    stop_words = set(stopwords.words('english'))
    tokens =word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

labels,vector,model = load_models()
st.title('Emotions Detected')
st.markdown('Enter the text and the emotions behind it will be predicted')

user_input = st.text_area("", placeholder="e.g. I feel so hopeless today...", height=130, label_visibility="collapsed")


clicked = st.button("Detect Emotion")
if clicked:
    if not user_input :
        st.warning("Please enter in the above box ⚠️")
    else:
        preprocessed_input = preprocessing(user_input)
        input_vector = vector.transform([preprocessed_input])
        output_labeled = model.predict(input_vector)[0]
        new_dic = {}
        for key , values in labels.items():
            new_dic[values]=key
        pred = new_dic[output_labeled]
        st.markdown(f"The statement is {pred}")