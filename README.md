# 🧠 Emotion Detector NLP

A text-based emotion classification app built with **scikit-learn** and **Streamlit**. Type any sentence and the model detects the emotion behind it.

## 🎯 Emotions Detected
`sadness` · `anger` · `love` · `surprise` · `fear` · `joy`

## 🚀 Live Demo
> Coming soon on Streamlit Cloud

## 🛠️ Tech Stack
- **Model:** Logistic Regression
- **Vectorizer:** TF-IDF
- **NLP Preprocessing:** NLTK (stopword removal, tokenization)
- **UI:** Streamlit
- **Model Persistence:** Joblib

## 📁 Project Structure
```
emotion-detector-nlp/
├── app.py                  # Streamlit app
├── model_lr.pkl            # Trained Logistic Regression model
├── tfidf_vectorizer.pkl    # Fitted TF-IDF vectorizer
├── emotion_labels.pkl      # Emotion label mapping
├── requirements.txt        # Dependencies
└── README.md
```

## ⚙️ Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/priyanshu812/emotion-detector-nlp.git
cd emotion-detector-nlp
```

**2. Create and activate virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download NLTK data (first time only)**
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

**5. Run the app**
```bash
streamlit run app.py
```

## 📊 Model Performance

| Model | Vectorizer | Accuracy |
|-------|-----------|----------|
| Naive Bayes | BoW | 76.78% |
| Naive Bayes | TF-IDF | 66.09% |
| **Logistic Regression** | **TF-IDF** | **86.15%** ✅ |

## 🔮 Roadmap
- [ ] Add Deep Learning model (LSTM / DistilBERT)
- [ ] Model selection toggle in UI
- [ ] Deploy to Streamlit Cloud

## 👨‍💻 Author
**Priyanshu Soni** · [LinkedIn](https://linkedin.com/in/priyanshu-soni-ai) · [GitHub](https://github.com/priyanshu812)
