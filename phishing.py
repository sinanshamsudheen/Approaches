import pandas as pd
df=pd.read_csv("dataset_phishing.csv")
df['fake']=df.status.apply(lambda x: 1 if x=="phishing" else 0)
# print(df.shape)
# print(df.isna().sum().sum()) 
df2 = df[df['status'].isin(['legitimate', 'phishing'])]
# print(df2.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(df.url,df.fake,test_size=0.2,random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

model_params = {
      'svm': {
          'model': SVC(),
        'params': {
              'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }
    },
    'random_forest': {
          'model': RandomForestClassifier(),
        'params': {
              'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'criterion': ['gini', 'entropy']
        }
    },
    'logistic_regression': {
          'model': LogisticRegression(solver='liblinear'),
        'params': {
              'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2']
        }
    }
}
results = []

for name, mp in model_params.items():
    print(f"working on {name}..")
    search = RandomizedSearchCV(mp['model'], mp['params'], cv=5, n_iter=10, scoring='f1', n_jobs=-1)
    search.fit(X_train_vec, y_train)

    best_model = search.best_estimator_
    best_score = search.best_score_
    best_params = search.best_params_

    results.append({
            'model': name,
            'best_score': best_score,
            'best_params': best_params
        })

df_results = pd.DataFrame(results, columns=['model', 'best_score', 'best_params'])
df_results = df_results.sort_values(by='best_score', ascending=False).reset_index(drop=True)

df_results.to_csv("model_comparison_results.csv", index=False)