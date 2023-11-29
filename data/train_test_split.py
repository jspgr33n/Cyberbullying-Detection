import pandas as pd
from sklearn.model_selection import train_test_split

file_path = 'removed_combined_csv.csv' 
data = pd.read_csv(file_path)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

train_data.to_csv('train_data_removed.csv', index=False)
test_data.to_csv('test_data_removed.csv', index=False)

