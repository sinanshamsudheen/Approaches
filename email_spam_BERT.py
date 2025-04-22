import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import tensorflow as tf

# Load dataset
df=pd.read_csv("spam.csv")
df['spam']=df.Category.apply(lambda x: 1 if x=="spam" else 0)

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['Message'].tolist(), df['spam'].tolist(), test_size=0.25)

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
)).batch(16)

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    test_labels
)).batch(16)

# Load pre-trained BERT model
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

# Compile & Train
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

model.fit(train_dataset, epochs=3, validation_data=test_dataset)

# Evaluate
loss, acc = model.evaluate(test_dataset)
print(f"Test Accuracy: {acc * 100:.2f}%")
