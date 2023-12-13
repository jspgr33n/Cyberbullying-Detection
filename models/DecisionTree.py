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

# Assuming 'tweet_text' column contains preprocessed text and 'cyberbullying_type' column contains labels
df['tweet_text'] = df['tweet_text'].fillna('')

texts = df['tweet_text'].apply(lambda x: x.split())  # Splitting preprocessed text into words
labels = df['cyberbullying_type'].values

# Train Word2Vec model
word2vec = Word2Vec(sentences=texts, vector_size=100, window=5, min_count=1, workers=4)

# Convert texts to average Word2Vec vectors
def text_to_avg_vector(text):
    vectors = [word2vec.wv[word] for word in text if word in word2vec.wv]
    if not vectors:
        # Return a vector of zeros if the text has no valid words
        return np.zeros(word2vec.vector_size)
    return np.mean(vectors, axis=0)

X = np.array([text_to_avg_vector(text) for text in texts])
y = np.array(labels)

# Handling NaN values (if any)
X = np.nan_to_num(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Best parameters without performing grid search
best_params = {'max_depth': 27, 'min_samples_leaf': 12}

# Initialize Decision Tree classifier with the best parameters
best_model = DecisionTreeClassifier(**best_params)

# Fit the model on the training data
best_model.fit(X_train, y_train)

# Evaluate the model
predictions = best_model.predict(X_test)
print(classification_report(y_test, predictions))

# Use your best Decision Tree model for prediction
y_pred_prob = best_model.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

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
plt.title('ROC Curve for DT with Best Params')
plt.legend(loc='lower right')
plt.show()


# # Define the parameter grid for the grid search
# param_grid = {
#     'min_samples_leaf': range(1, 101),
#     'max_depth': range(1, 101)
# }

# # Initialize Decision Tree classifier
# dt_classifier = DecisionTreeClassifier()


# # Perform grid search using cross-validation with progress printing
# grid_search = GridSearchCV(dt_classifier, param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
# grid_search.fit(X_train, y_train)

# # Print the best parameters
# print("Best Parameters:", grid_search.best_params_)

# # Evaluate the model with the best parameters
# best_model = grid_search.best_estimator_
# predictions = best_model.predict(X_test)
# print(classification_report(y_test, predictions))
