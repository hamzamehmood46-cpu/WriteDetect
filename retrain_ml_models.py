"""Retrain ML models using the current sklearn version in this venv."""
import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import re
import nltk

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Download required NLTK data quietly
for r in ['stopwords', 'punkt', 'punkt_tab', 'wordnet', 'omw-1.4',
          'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
    try:
        nltk.download(r, quiet=True)
    except Exception:
        pass

# Load data
DATA_FILE = r"C:\Users\hamza\Downloads\Class_work\Intro To Ai Agents\Muhammad_Haseeb_project_2\data\train_data with labels (2).xlsx"
print(f"Loading data from {DATA_FILE}...")
df = pd.read_excel(DATA_FILE)
df.columns = ['essay', 'label']
df = df.dropna(subset=['essay', 'label'])
df['label'] = df['label'].astype(int)
print(f"Loaded {len(df)} samples. Label distribution:\n{df['label'].value_counts()}")

# Simple preprocessing (same as notebook)
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def simple_preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

print("Preprocessing texts...")
df['processed'] = df['essay'].apply(simple_preprocess)

X = df['processed'].values
y = df['label'].values

# TF-IDF vectorizer (shared config)
tfidf_params = dict(ngram_range=(1, 2), max_features=50000, min_df=2, sublinear_tf=True)

# 1. SVM Pipeline
print("Training SVM...")
pipeline_svm = Pipeline([
    ('vectorizer', TfidfVectorizer(**tfidf_params)),
    ('model', SVC(kernel='linear', probability=True, random_state=42))
])
pipeline_svm.fit(X, y)
joblib.dump(pipeline_svm, 'models/svm_pipeline.pkl')
print("  Saved models/svm_pipeline.pkl")

# 2. Decision Tree Pipeline
print("Training Decision Tree...")
pipeline_dt = Pipeline([
    ('vectorizer', TfidfVectorizer(**tfidf_params)),
    ('model', DecisionTreeClassifier(max_depth=20, min_samples_split=5, random_state=42))
])
pipeline_dt.fit(X, y)
joblib.dump(pipeline_dt, 'models/decision_tree_pipeline.pkl')
print("  Saved models/decision_tree_pipeline.pkl")

# 3. AdaBoost Pipeline
print("Training AdaBoost...")
pipeline_ada = Pipeline([
    ('vectorizer', TfidfVectorizer(**tfidf_params)),
    ('model', AdaBoostClassifier(n_estimators=100, random_state=42))
])
pipeline_ada.fit(X, y)
joblib.dump(pipeline_ada, 'models/adaboost_pipeline.pkl')
print("  Saved models/adaboost_pipeline.pkl")

print("\nAll ML models retrained and saved successfully!")
print("Restart the Streamlit app to use the new models.")
