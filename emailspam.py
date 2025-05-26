from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import pandas as pd

df = pd.read_csv('spam.csv', encoding='latin-1')[['Category', 'Message']]
df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# Use fewer rows for lower RAM usage (weak pc here)
df = df.head(500)

X = df['Message']
y = df['spam']

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
encodings = tokenizer(X, truncation=True, padding=True, return_tensors='tf')

model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

optimizer = Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
model.fit(encodings['input_ids'], np.array(y), epochs=3)
