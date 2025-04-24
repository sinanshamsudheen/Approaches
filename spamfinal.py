import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from transformers import AutoTokenizer, AutoModel
import torch

# --- 1. Load and preprocess data ---
df = pd.read_csv('spam.csv', encoding='latin-1')[['Category', 'Message']]
df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# Use fewer rows for lower RAM usage (optional)
df = df.head(500)

X = df['Message']
y = df['spam']

# --- 2. Custom BERT Transformer ---
class BERTTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='distilbert-base-uncased'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        embeddings = []
        for text in X:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(cls_embedding)
        return np.array(embeddings)

# --- 3. Custom Combined Pipeline ---
class CustomPipeline(BaseEstimator):
    def __init__(self, model_name='distilbert-base-uncased'):
        self.bert_pipe = BERTTransformer(model_name)
        self.tfidf_pipe = TfidfVectorizer(max_features=300)
        self.classifier = LogisticRegression()

    def fit(self, X, y):
        bert_features = self.bert_pipe.fit_transform(X)
        tfidf_features = self.tfidf_pipe.fit_transform(X).toarray()
        combined = np.hstack([bert_features, tfidf_features])
        self.classifier.fit(combined, y)
        return self

    def predict(self, X):
        bert_features = self.bert_pipe.transform(X)
        tfidf_features = self.tfidf_pipe.transform(X).toarray()
        combined = np.hstack([bert_features, tfidf_features])
        return self.classifier.predict(combined)

# --- 4. Train and evaluate ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = CustomPipeline()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
