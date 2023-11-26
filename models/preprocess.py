import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

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

    # Join the cleaned words back into a single string
    cleaned_text = ' '.join(words)

    return cleaned_text

# Apply text preprocessing to the 'tweet_text' column and create a new 'text' column
df['tweet_text'] = df['tweet_text'].apply(preprocess_text)

# Changing cyberbullying_type to binary variable (0 for non-CB, 1 for others)
df['cyberbullying_type'] = df['cyberbullying_type'].apply(lambda x: 0 if x == 'not_cyberbullying' else 1)

# Save the updated DataFrame to a new CSV file
output_csv_file = os.path.join(data_folder, 'updated_cyberbullying_data.csv')
df.to_csv(output_csv_file, index=False)

# Split the dataset into training and testing sets
X = df['tweet_text']  # Text data
y = df['cyberbullying_type']  # Labels (0 for non-cyberbullying, 1 for cyberbullying)

# Split data into 70% training and 30% testing
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, random_state=334)

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust the number of features as needed
X_train_tfidf = tfidf_vectorizer.fit_transform(xTrain)
X_test_tfidf = tfidf_vectorizer.transform(xTest)