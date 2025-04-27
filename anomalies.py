import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv('cybersec_anom_data.csv')
df.dropna(inplace=True)
print(df.isnull().sum())
print(df.shape)

numeric_features = ['network_packet_size', 'login_attempts', 'session_duration', 
                   'ip_reputation_score', 'failed_logins', 'unusual_time_access']
categorical_features = ['protocol_type', 'encryption_used', 'browser_type']

df['login_frequency'] = df['login_attempts'] / (df['session_duration'] + 1) 

def calculate_entropy(row):
    entropy_score = (
        (row['login_attempts'] / 10) * 
        (1 / (row['ip_reputation_score'] + 1)) * 
        (row['failed_logins'] + 1) * 
        (2 if row['unusual_time_access'] > 0 else 1)
    )
    return min(entropy_score, 1.0)  

df['session_entropy'] = df.apply(calculate_entropy, axis=1)

numeric_features.append('session_entropy')
numeric_features.append('login_frequency')

X = df.drop(['session_id', 'attack_detected'], axis=1)
y_true = df['attack_detected']  

X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2, random_state=42)

contamination_rate = y_true.mean()
print(f"Actual anomaly rate in dataset: {contamination_rate:.4f}")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
isolation_forest = IsolationForest(
    n_estimators=100,      
    max_samples=256,       
    contamination=contamination_rate,  
    random_state=42
)
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('isolation_forest', isolation_forest)
])

pipeline.fit(X_train)
y_pred = np.where(pipeline.predict(X_test) == -1, 1, 0)
print(classification_report(y_test, y_pred))
