import os
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from gensim.models import Word2Vec  # Use Word2Vec for training word embeddings

# Get the current working directory
current_directory = os.getcwd()

# Move up one level to the parent folder
parent_directory = os.path.dirname(current_directory)

# Define the path to the data folder and the dataset file
data_folder = os.path.join(parent_directory, 'data')
data_file = os.path.join(data_folder, 'cyberbullying_data.csv')

# Load the dataset
df = pd.read_csv(data_file)

# Check for and handle missing values
df['tweet_text'].fillna('', inplace=True)  # Replace NaN values with empty strings

# Text preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Tokenization (split text into words)
    # nltk is a package that allows users to access tools such as tokenizers, lists of stopwords, etc.
    words = nltk.word_tokenize(text)

    # Remove special characters, numbers, and punctuation
    words = [re.sub(r'[^a-zA-Z]', '', word) for word in words if word.isalpha()]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    return words

# Apply text preprocessing to the 'tweet_text' column and create a new 'tokens' column
df['tokens'] = df['tweet_text'].apply(preprocess_text)

# Set 'cyberbullying_type' to 0 for 'not_cyberbullying' and 1 for everything else
df['cyberbullying_type'] = df['cyberbullying_type'].apply(lambda x: 0 if x == 'not_cyberbullying' else 1)

# Split the dataset into training and testing sets
X = df['tokens']  # Text data
y = df['cyberbullying_type']  # Labels (0 for not_cyberbullying, 1 for cyberbullying)

# Split data into 80% training and 20% testing
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=334)

# Train Word2Vec word embeddings on the training set
word2vec_model = Word2Vec(xTrain, vector_size=100, window=5, min_count=1, sg=0)  # Adjust parameters as needed

# Function to get document embeddings
def get_document_embedding(tokens, model):
    # Filter tokens that are present in the Word2Vec model's vocabulary
    valid_tokens = [token for token in tokens if token in model.wv.index_to_key]
    
    # If no valid tokens are found, return a zero vector
    if not valid_tokens:
        return np.zeros(model.vector_size)
    
    # Calculate the mean of word vectors for the valid tokens
    embeddings = [model.wv[token] for token in valid_tokens]
    doc_embedding = np.mean(embeddings, axis=0)
    
    return doc_embedding

# Apply the function to create document embeddings
xTrain_embeddings = xTrain.apply(lambda x: get_document_embedding(x, word2vec_model))
xTest_embeddings = xTest.apply(lambda x: get_document_embedding(x, word2vec_model))

# Save the updated DataFrame to a new CSV file (if needed)
output_csv_file = os.path.join(data_folder, 'updated_cyberbullying_data_word2vec_embedding.csv')
df.to_csv(output_csv_file, index=False)
