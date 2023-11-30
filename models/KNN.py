import os
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    classification_report, 
    roc_curve, 
    auc, 
    precision_recall_curve, 
    average_precision_score
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

# Initialize lists to store evaluation metrics and accuracy scores
accuracy_scores = []
f1_scores = []
roc_aucs = []
pr_aucs = []  # Added list to store area under precision-recall curve values
auroc_scores = []  # Added list to store AUROC values

# Initialize variables to track the best k, its corresponding F1 score, and accuracy
best_k = None
best_f1 = -1.0
best_accuracy = -1.0
best_pr_auc = -1.0  # Variable to track the best AUC-PRC
best_k_index = -1  # Index of the best k value

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
        best_k_index = i
    
    # Calculate precision-recall curve and AUC-PRC for class 0 (non-cyberbullying)
    precision_0, recall_0, _ = precision_recall_curve(yTest, knn_classifier.predict_proba(xTest)[:, 0], pos_label=0)
    pr_auc0 = auc(recall_0, precision_0)
    
    # Calculate precision-recall curve and AUC-PRC for class 1 (cyberbullying)
    precision_1, recall_1, _ = precision_recall_curve(yTest, knn_classifier.predict_proba(xTest)[:, 1], pos_label=1)
    pr_auc1 = auc(recall_1, precision_1)
    pr_aucs.append((pr_auc0, pr_auc1))
    
    # Calculate ROC curve and AUROC
    fpr, tpr, _ = roc_curve(yTest, knn_classifier.predict_proba(xTest)[:, 1])
    auroc = auc(fpr, tpr)
    auroc_scores.append(auroc)
    
    # Update best_pr_auc if a better AUC-PRC is found for class 1
    if pr_auc1 > best_pr_auc:
        best_pr_auc = pr_auc1
    
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

# Plot F1 score, AUC-PRC, and AUROC for different k values
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.plot(k_values, f1_scores, marker='o', linestyle='-', label='F1 Score')
plt.xlabel('K Value')
plt.ylabel('F1 Score')
odd_k_values = [k for k in k_values if k % 2 == 1]
plt.xticks(odd_k_values)  # Set x-axis ticks to include only odd k values
plt.title('F1 Score vs. K Value')
plt.grid(True)

plt.subplot(1, 3, 2)
# Plot precision-recall curves for both classes (0 and 1)
precision_0_best = precision_0[best_k_index]
recall_0_best = recall_0[best_k_index]
pr_auc_0_best = pr_aucs[best_k_index][0]

precision_1_best = precision_1[best_k_index]
recall_1_best = recall_1[best_k_index]
pr_auc_1_best = pr_aucs[best_k_index][1]

plt.plot(recall_0_best, precision_0_best, marker='o', linestyle='-', label='Precision-Recall (Class 0)')
plt.plot(recall_1_best, precision_1_best, marker='o', linestyle='-', label='Precision-Recall (Class 1)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Best Model')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
# Plot AUROC for different k values
plt.plot(k_values, auroc_scores, marker='o', linestyle='-', label='AUROC')
plt.xlabel('K Value')
plt.ylabel('AUROC')
plt.xticks(odd_k_values)  # Set x-axis ticks to include only odd k values
plt.title('AUROC vs. K Value')
plt.grid(True)

plt.tight_layout()
plt.show()

# Display the best k value and evaluation metrics
print(f"Best K Value (Based on F1 Score): {best_k}")
print(f"F1 Score: {best_f1:.2f}")
print(f"Accuracy: {best_accuracy:.2f}")
print(f"Best AUC-PRC (Class 1): {best_pr_auc:.2f}")
print(f"Classification Report:\n{classification_rep}")

# Plot the ROC curve for the best model
plt.figure()
plt.plot(fpr_best, tpr_best, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc_best))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for KNN with Best K-value')
plt.legend(loc='lower right')
plt.show()

# Initialize lists to store accuracy scores
accuracy_scores = []

# Train KNN models for different odd k values and evaluate them
for k in k_values:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
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
