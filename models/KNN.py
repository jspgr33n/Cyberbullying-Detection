import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    roc_curve, 
    auc
)
from gensim.models import Word2Vec
import matplotlib.pyplot as plt

# Get the current working directory, should be within models folder for code to work properly
current_directory = os.getcwd()

# Move up one level to the parent folder
parent_directory = os.path.dirname(current_directory)

# Define the path to the data folder and the dataset file
data_folder = os.path.join(parent_directory, 'data')
data_file = os.path.join(data_folder, 'removed_knn_csv.csv')
df = pd.read_csv(os.path.join(data_folder, data_file))

X = df['tokens']  
y = df['cyberbullying_type']

word2vec_model = Word2Vec(sentences=X, vector_size=100, window=5, min_count=1, sg=0) 

def get_document_embedding(tokens, model):
    valid_tokens = [token for token in tokens if token in model.wv.key_to_index]
    if not valid_tokens:
        return np.zeros(model.vector_size)    
    embeddings = [model.wv[token] for token in valid_tokens]
    doc_embedding = np.mean(embeddings, axis=0)   
    return doc_embedding

df['document_embeddings'] = df['tokens'].apply(lambda x: get_document_embedding(x, word2vec_model))

# Split data into training and testing sets (80% training, 20% testing)
xTrain, xTest, yTrain, yTest = train_test_split(df['document_embeddings'].tolist(), y, test_size=0.2, random_state=334)

k_values = list(range(1, 31, 2))

accuracy_scores = []

knn_classifier = KNeighborsClassifier()

param_grid = {'n_neighbors': k_values, 'weights': ['uniform', 'distance']}

grid_search = GridSearchCV(knn_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(xTrain, yTrain)

best_k = grid_search.best_params_['n_neighbors']
best_weight = grid_search.best_params_['weights']

best_knn_classifier = KNeighborsClassifier(n_neighbors=best_k, weights=best_weight)
best_knn_classifier.fit(xTrain, yTrain)

y_pred_best = best_knn_classifier.predict(xTest)
classification_rep = classification_report(yTest, y_pred_best)

for k in k_values:
    knn_classifier = KNeighborsClassifier(n_neighbors=k, weights=best_weight)
    knn_classifier.fit(xTrain, yTrain)
    y_pred = knn_classifier.predict(xTest)
    accuracy = accuracy_score(yTest, y_pred)
    accuracy_scores.append(accuracy)

plt.figure(figsize=(8, 6))
plt.plot(k_values, accuracy_scores, marker='o', linestyle='-')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. K Value (KNN)')
plt.grid(True)
plt.show()

print(f"Best K Value (Based on Accuracy): {best_k}")
print(f"Best Weight: {best_weight}")
print(f"Classification Report:\n{classification_rep}")
