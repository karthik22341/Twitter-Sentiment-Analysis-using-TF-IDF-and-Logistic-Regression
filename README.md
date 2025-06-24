# Twitter-Sentiment-Analysis-using-TF-IDF-and-Logistic-Regression
A Twitter Sentiment Analysis project using Python, NLP preprocessing (NLTK), TF-IDF vectorization, and Logistic Regression. Tweets are classified into sentiment categories with accuracy evaluation and a clear machine learning pipeline.


# Twitter Sentiment Analysis using TF-IDF and Logistic Regression

This project performs sentiment analysis on tweets using Natural Language Processing (NLP) and a machine learning model. It includes text preprocessing, feature extraction using TF-IDF, and sentiment classification using Logistic Regression.

---

## 🚀 Features
- Data cleaning using `re` and `nltk` (stopword removal, stemming)
- Feature extraction with `TfidfVectorizer`
- Classification using `LogisticRegression`
- Accuracy evaluation with `sklearn.metrics`
- Clean modular code for preprocessing and training

---

## 📦 Libraries Used
- `pandas` – Data manipulation
- `numpy` – Array operations
- `re` – Regular expressions for text cleaning
- `nltk` – Stopword removal and stemming
- `scikit-learn` – TF-IDF, model training, evaluation

---

## 🧠 ML Workflow

1. **Data Preprocessing**:
   - Lowercasing, removing special characters
   - Stopword removal (NLTK)
   - Stemming using `PorterStemmer`

2. **Vectorization**:
   - Using `TfidfVectorizer` to convert text into numerical vectors

3. **Model Training**:
   - Train/test split using `train_test_split`
   - Model: `LogisticRegression`

4. **Evaluation**:
   - Accuracy score on test data

---

## 📊 Example Prediction
Given a tweet like:

> `"I'm so happy with the new update!"`

The model predicts: `Positive`

---

## 📁 Folder Structure (If You Have Web App or UI)
```bash
project/
│
├── app.py              # Flask web app (if applicable)
├── model.joblib        # Trained model
├── utils.py            # Preprocessing functions (optional)
├── templates/
│   └── index.html      # Web interface
├── static/
│   └── style.css       # Styling
└── README.md
