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

# Split the dataset into features (X) and labels (y)
X = df['tokens']  # Tokenized text data
y = df['cyberbullying_type']

# Train Word2Vec model on the training data
word2vec_model = Word2Vec(sentences=X, vector_size=100, window=5, min_count=1, sg=0)  # You can adjust parameters as needed

# Function to get document embeddings
def get_document_embedding(tokens, model):
    # Filter tokens that are present in the Word2Vec model's vocabulary
    valid_tokens = [token for token in tokens if token in model.wv.key_to_index]
    
    # If no valid tokens are found, return a zero vector
    if not valid_tokens:
        return np.zeros(model.vector_size)
    
    # Calculate the mean of word vectors for the valid tokens
    embeddings = [model.wv[token] for token in valid_tokens]
    doc_embedding = np.mean(embeddings, axis=0)
    
    return doc_embedding

# Apply the function to create document embeddings
df['document_embeddings'] = df['tokens'].apply(lambda x: get_document_embedding(x, word2vec_model))

# Split data into training and testing sets (80% training, 20% testing)
xTrain, xTest, yTrain, yTest = train_test_split(df['document_embeddings'].tolist(), y, test_size=0.2, random_state=334)

# Define a range of k values for KNN
k_values = list(range(1, 31, 2))

# Initialize lists to store accuracy scores
accuracy_scores = []

# Create a K-nearest neighbors classifier
knn_classifier = KNeighborsClassifier()

# Define a grid of hyperparameters to search
param_grid = {'n_neighbors': k_values, 'weights': ['uniform', 'distance']}

# Create a grid search with cross-validation
grid_search = GridSearchCV(knn_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(xTrain, yTrain)

# Get the best hyperparameters from the grid search
best_k = grid_search.best_params_['n_neighbors']
best_weight = grid_search.best_params_['weights']

# Train the best KNN model with the best k value and weight
best_knn_classifier = KNeighborsClassifier(n_neighbors=best_k, weights=best_weight)
best_knn_classifier.fit(xTrain, yTrain)

# Evaluate the best model and calculate accuracy
y_pred_best = best_knn_classifier.predict(xTest)
classification_rep = classification_report(yTest, y_pred_best)

# Calculate accuracy for different k values
for k in k_values:
    knn_classifier = KNeighborsClassifier(n_neighbors=k, weights=best_weight)
    knn_classifier.fit(xTrain, yTrain)
    y_pred = knn_classifier.predict(xTest)
    
    # Calculate accuracy
    accuracy = accuracy_score(yTest, y_pred)
    accuracy_scores.append(accuracy)

# Plot accuracy for different k values
plt.figure(figsize=(8, 6))
plt.plot(k_values, accuracy_scores, marker='o', linestyle='-')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. K Value (KNN)')
plt.grid(True)
plt.show()

# Print the best K value, weight, and classification report
print(f"Best K Value (Based on Accuracy): {best_k}")
print(f"Best Weight: {best_weight}")
print(f"Classification Report:\n{classification_rep}")
