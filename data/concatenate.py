import pandas as pd

features_df = pd.read_csv('raw_tweets.csv')  
labels_df = pd.read_csv('classification.csv') 

combined_df = pd.concat([features_df, labels_df], axis=1)

combined_df.to_csv('combined_csv.csv', index=False)
