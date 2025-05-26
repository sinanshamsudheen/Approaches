from transformers import BertTokenizer, BertModel
from catboost import CatBoostClassifier
import torch
import numpy as np

# Simulated email/message contents
texts = ["Quarterly revenue summary attached", "Lunch at 1?", "Confidential contract details inside"]
behaviors = np.array([[0, 1, 5], [1, 0, 2], [0, 1, 9]])  # [time_of_day, to_public, data_volume]
labels = [1, 0, 1]

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

with torch.no_grad():
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    content_embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # CLS token

# Combine BERT embeddings with behavioral features
X = np.hstack([content_embeddings, behaviors])

clf = CatBoostClassifier(verbose=0)
clf.fit(X, labels)

# Predict leak risk
print("Predicted Risk Classes:", clf.predict(X))
