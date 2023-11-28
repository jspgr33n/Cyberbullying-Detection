import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv('../data/updated_cyberbullying_data.csv')

# Handle NaN values in 'tweet_text'
df['tweet_text'] = df['tweet_text'].fillna('')

# Splitting strings into words
texts = df['tweet_text'].apply(lambda x: x.split())
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

# Apply the function to each text
X = np.array([text_to_avg_vector(text) for text in texts])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train a Logistic Regression classifier
model = LogisticRegression(max_iter=1000)  # Adjust max_iter based on your dataset size
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
