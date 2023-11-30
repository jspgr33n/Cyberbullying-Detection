import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from gensim.models import Word2Vec
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import classification_report, roc_curve, auc


# Get the current working directory, should be within models folder for code to work properly
current_directory = os.getcwd()

# Move up one level to the parent folder
parent_directory = os.path.dirname(current_directory)

# Define the path to the data folder and the dataset file
data_folder = os.path.join(parent_directory, 'data')
data_file = os.path.join(data_folder, 'removed_knn_csv.csv')
df = pd.read_csv(os.path.join(data_folder, data_file))

# Split the data into training and testing sets
xTrain, xTest, yTrain, yTest = train_test_split(df['tweet_text'], df['cyberbullying_type'], test_size=0.2, random_state=334)

# Train a Word2Vec model on your text data (you can adjust parameters as needed)
word2vec_model = Word2Vec(sentences=xTrain, vector_size=100, window=5, min_count=1, sg=0)

# Function to compute the document embedding from word embeddings
def compute_doc_embedding(text, model):
    words = text.split()
    valid_words = [word for word in words if word in model.wv.key_to_index]
    if not valid_words:
        return np.zeros(model.vector_size)
    word_vectors = [model.wv[word] for word in valid_words]
    doc_embedding = np.mean(word_vectors, axis=0)
    return doc_embedding

# Convert text data to document embeddings using the trained Word2Vec model
xTrain_embeddings = [compute_doc_embedding(text, word2vec_model) for text in xTrain]
xTest_embeddings = [compute_doc_embedding(text, word2vec_model) for text in xTest]

# Define hyperparameters for grid search
# param_grid = {
#     'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
#     'penalty': ['l1', 'l2'],  # Penalty type (L1 or L2)
#     'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'],  # Solver
#     'class_weight': [None, 'balanced'],  # Class weights
#     'max_iter': [100]  # Maximum iterations
# }

# # Create a Logistic Regression model
# logistic_regression = LogisticRegression()

# # Perform grid search with cross-validation and progress printing
# grid_search = GridSearchCV(estimator=logistic_regression, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)

# # Fit the grid search to your training data
# grid_search.fit(xTrain_embeddings, yTrain)

# Get the best hyperparameters
best_params = {'C': 100, 'class_weight': None, 'max_iter': 100, 'penalty': 'l2', 'solver': 'liblinear'}

# Create a logistic regression model with the best hyperparameters
best_logistic_regression = LogisticRegression(**best_params)

# Train the best model on the document embeddings of the training data
best_logistic_regression.fit(xTrain_embeddings, yTrain)

# Make predictions on the document embeddings of the testing data
y_pred = best_logistic_regression.predict(xTest_embeddings)

# Evaluate the best model
accuracy = accuracy_score(yTest, y_pred)
classification_rep = classification_report(yTest, y_pred)

# Display evaluation metrics and classification report
print(f"Best Hyperparameters: {best_params}")
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_rep)


# Train the best model on the document embeddings of the training data
best_logistic_regression.fit(xTrain_embeddings, yTrain)

# Make predictions on the document embeddings of the testing data
y_pred = best_logistic_regression.predict(xTest_embeddings)

# Evaluate the best model
accuracy = accuracy_score(yTest, y_pred)
classification_rep = classification_report(yTest, y_pred)

# Display evaluation metrics and classification report
print(f"Best Hyperparameters: {best_params}")
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_rep)

# Calculate the probabilities for positive class (class 1)
y_pred_prob = best_logistic_regression.predict_proba(xTest_embeddings)[:, 1]

# Compute ROC curve
fpr, tpr, _ = roc_curve(yTest, y_pred_prob)

# Calculate AUROC
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for LR with Best Params')
plt.legend(loc='lower right')
plt.show()