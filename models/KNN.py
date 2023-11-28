import os
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_curve, auc
from gensim.models import Word2Vec
import matplotlib.pyplot as plt

# Get the current working directory, should be within models folder for code to work properly
current_directory = os.getcwd()

# Move up one level to the parent folder
parent_directory = os.path.dirname(current_directory)

# Define the path to the data folder and the dataset file
data_folder = os.path.join(parent_directory, 'data')
data_file = os.path.join(data_folder, 'updated_cyberbullying_data_word2vec_embedding.csv')
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
k_values = list(range(1, 31))

# Initialize lists to store evaluation metrics and accuracy scores
accuracy_scores = []
f1_scores = []
roc_aucs = []

# Initialize variables to track the best k, its corresponding F1 score, and accuracy
best_k = None
best_f1 = -1.0
best_accuracy = -1.0

# Train KNN models for different k values and evaluate them
for i, k in enumerate(k_values):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(xTrain, yTrain)
    y_pred = knn_classifier.predict(xTest)
    
    # Calculate F1 score
    f1 = f1_score(yTest, y_pred)
    f1_scores.append(f1)
    
    # Calculate accuracy
    accuracy = accuracy_score(yTest, y_pred)
    accuracy_scores.append(accuracy)
    
    # Update best_k and best_f1 if a better F1 score is found
    if f1 > best_f1:
        best_k = k
        best_f1 = f1
        best_accuracy = accuracy
    
    # Print progress and percentage completion
    progress = (i + 1) / len(k_values) * 100
    print(f"Progress: {progress:.2f}%")

# Train the KNN model with the best k value based on F1 score
best_knn_classifier = KNeighborsClassifier(n_neighbors=best_k)
best_knn_classifier.fit(xTrain, yTrain)

# Evaluate the best model and calculate F1 score and AUROC
y_pred_best = best_knn_classifier.predict(xTest)
classification_rep = classification_report(yTest, y_pred_best)

fpr_best, tpr_best, _ = roc_curve(yTest, best_knn_classifier.predict_proba(xTest)[:,1])
roc_auc_best = auc(fpr_best, tpr_best)

# Plot F1 score for different k values
plt.figure(figsize=(6, 4))
plt.plot(k_values, f1_scores, marker='o', linestyle='-', label='F1 Score')
plt.plot(k_values, accuracy_scores, marker='o', linestyle='-', label='Accuracy')  # Adding accuracy to the plot
plt.xlabel('K Value')
plt.ylabel('Score')
plt.xticks(k_values, 5)
plt.title('F1 Score and Accuracy vs. K Value')
plt.legend()
plt.show()

# Display the best k value and evaluation metrics
print(f"Best K Value (Based on F1 Score): {best_k}")
print(f"F1 Score: {best_f1:.2f}")
print(f"Accuracy: {best_accuracy:.2f}")
print(f"Classification Report:\n{classification_rep}")

# Plot the ROC curve for the best model
plt.figure()
plt.plot(fpr_best, tpr_best, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc_best))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Best Model)')
plt.legend(loc='lower right')
plt.show()
