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

data_folder = os.path.join(parent_directory, 'data')
data_file = os.path.join(data_folder, 'removed_knn_csv.csv')
df = pd.read_csv(os.path.join(data_folder, data_file))

xTrain, xTest, yTrain, yTest = train_test_split(df['tweet_text'], df['cyberbullying_type'], test_size=0.2, random_state=334)

word2vec_model = Word2Vec(sentences=xTrain, vector_size=100, window=5, min_count=1, sg=0)

def compute_doc_embedding(text, model):
    words = text.split()
    valid_words = [word for word in words if word in model.wv.key_to_index]
    if not valid_words:
        return np.zeros(model.vector_size)
    word_vectors = [model.wv[word] for word in valid_words]
    doc_embedding = np.mean(word_vectors, axis=0)
    return doc_embedding

xTrain_embeddings = [compute_doc_embedding(text, word2vec_model) for text in xTrain]
xTest_embeddings = [compute_doc_embedding(text, word2vec_model) for text in xTest]

# define a range of C values (learning rates) for logistic regression
c_values = [0.001, 0.01, 0.1, 1, 10, 100]

accuracy_scores = []

for c in c_values:
    logistic_regression = LogisticRegression(C=c, class_weight=None, max_iter=100, solver='liblinear', penalty='l2')
    logistic_regression.fit(xTrain_embeddings, yTrain)
    y_pred = logistic_regression.predict(xTest_embeddings)
    
    accuracy = accuracy_score(yTest, y_pred)
    accuracy_scores.append(accuracy)

plt.figure(figsize=(8, 6))
plt.semilogx(c_values, accuracy_scores, marker='o', linestyle='-')
plt.xlabel('C Value (Log Scale)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Learning Rate (Logistic Regression)')
plt.grid(True)
plt.show()

# find best learning rate (C) for graph
best_c = c_values[np.argmax(accuracy_scores)]

best_params = {'C': best_c, 'class_weight': None, 'max_iter': 100, 'penalty': 'l2', 'solver': 'liblinear'}

best_logistic_regression = LogisticRegression(**best_params)

best_logistic_regression.fit(xTrain_embeddings, yTrain)

y_pred = best_logistic_regression.predict(xTest_embeddings)

accuracy = accuracy_score(yTest, y_pred)
classification_rep = classification_report(yTest, y_pred)

print(f"Best Hyperparameters: {best_params}")
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_rep)

y_pred_prob = best_logistic_regression.predict_proba(xTest_embeddings)[:, 1]
fpr, tpr, _ = roc_curve(yTest, y_pred_prob)
roc_auc = auc(fpr, tpr)

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