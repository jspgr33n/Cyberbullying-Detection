import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('../../data/removed_knn_csv.csv')
print(df.head)

# Assuming 'tweet_text' column contains preprocessed text and 'label' column contains labels
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

# Initialize lists to store accuracies
depth_accuracies = []
leaf_accuracies = []

# Iterating over max depths
for depth in range(3, 51, 3):
    clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=22)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    depth_accuracies.append(accuracy_score(y_test, y_pred))

# Iterating over min samples leaf
for leaf in range(3, 101, 3):
    clf = DecisionTreeClassifier(max_depth=9, min_samples_leaf=leaf)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    leaf_accuracies.append(accuracy_score(y_test, y_pred))

# Plotting Accuracy vs Max Depth
plt.figure(figsize=(10, 6))
plt.plot(range(3, 51, 3), depth_accuracies, marker='o')
plt.title('Accuracy vs Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

# Plotting Accuracy vs Min Samples Leaf
plt.figure(figsize=(10, 6))
plt.plot(range(3, 101, 3), leaf_accuracies, marker='o')
plt.title('Accuracy vs Min Samples Leaf')
plt.xlabel('Min Samples Leaf')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
