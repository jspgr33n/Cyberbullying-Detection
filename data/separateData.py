import pandas as pd

def main():
    df = pd.read_csv('cyberbullying_data.csv')
    
    # Splitting the features and classification
    preprocessed_tweets = df.drop('cyberbullying_type', axis=1)
    classification = df['cyberbullying_type']

    preprocessed_tweets.to_csv('raw_tweets.csv', index=False)
    #classification.to_csv('classification.csv', index=False)
    
    # Print the features and classification
    print("Features:")
    print(preprocessed_tweets)
    print("Classification:")
    print(classification)

if __name__ == '__main__':
    main()
