import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load the dataset
df = pd.read_csv('../data/removed_knn_csv.csv')

df['tweet_text'] = df['tweet_text'].fillna('')

texts = df['tweet_text'].apply(lambda x: x.split()) 
labels = df['cyberbullying_type'].values

word2vec = Word2Vec(sentences=texts, vector_size=100, window=5, min_count=1, workers=4)

def text_to_avg_vector(text):
    vectors = [word2vec.wv[word] for word in text if word in word2vec.wv]
    if not vectors:
        return np.zeros(word2vec.vector_size)
    return np.mean(vectors, axis=0)

X = np.array([text_to_avg_vector(text) for text in texts])
y = np.array(labels)

X = np.nan_to_num(X)

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Best parameters we got when we performed grid search
best_params = {'max_depth': 27, 'min_samples_leaf': 12}
best_model = DecisionTreeClassifier(**best_params)
best_model.fit(X_train, y_train)

predictions = best_model.predict(X_test)
print(classification_report(y_test, predictions))

y_pred_prob = best_model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for DT with Best Params')
plt.legend(loc='lower right')
plt.show()
