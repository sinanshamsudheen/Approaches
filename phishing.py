import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import re
import datetime

# Sample dataset
data = pd.DataFrame({
    'url': ["http://g00gle-login.com", "https://www.google.com"],
    'domain_registration_date': [
        '2025-04-01', '2000-09-15'
    ],
    'tls_issuer': ['Untrusted CA', 'Google Trust Services LLC'],
    'label': [1, 0]
})

# Homoglyph detector (simple example)
def homoglyph_density(url):
    homoglyphs = {'0': 'o', '1': 'l', '3': 'e', '5': 's', 'а': 'a', 'р': 'p'}  # Cyrillic + common tricks
    return sum(url.count(h) for h in homoglyphs) / len(url)

# Time since registration
def domain_age_in_days(registration_date):
    reg_date = datetime.datetime.strptime(registration_date, '%Y-%m-%d')
    return (datetime.datetime.now() - reg_date).days

# Mock TLS reputation scoring
trusted_issuers = ['Google Trust Services LLC', 'DigiCert Inc', 'Let\'s Encrypt']
def tls_reputation(issuer):
    return 1 if issuer in trusted_issuers else 0

# Feature Extractor
class URLFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_feat = pd.DataFrame()
        X_feat['homoglyph_density'] = X['url'].apply(homoglyph_density)
        X_feat['domain_age'] = X['domain_registration_date'].apply(domain_age_in_days)
        X_feat['tls_reputation'] = X['tls_issuer'].apply(tls_reputation)
        return X_feat

# N-gram similarity using CountVectorizer (mock for Alexa-like analysis)
class NgramVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self): self.vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=100)
    def fit(self, X, y=None): self.vectorizer.fit(X['url']); return self
    def transform(self, X): return self.vectorizer.transform(X['url']).toarray()

# Combine features
extractor = URLFeatureExtractor()
ngrammer = NgramVectorizer()

# Extract features
url_lexical = ngrammer.fit_transform(data)
url_contextual = extractor.transform(data)

# Final feature set
X_final = np.hstack([url_lexical, url_contextual])
y = data['label'].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# XGBoost model with class imbalance handling
model = xgb.XGBClassifier(scale_pos_weight=3, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Evaluate with Precision@99% Recall
y_scores = model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# Find precision at recall >= 0.99
for p, r, t in zip(precision, recall, thresholds):
    if r >= 0.99:
        print(f"Precision@99% Recall: {p:.2f}, Threshold: {t:.4f}")
        break
