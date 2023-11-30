from transformers import BertForSequenceClassification, BertTokenizer

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load an untrained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Load and tokenize test dataset
file_path = '../data/test_data_removed.csv' 
dataset = load_dataset('csv', data_files={'test': file_path})
tokenized_test = dataset.map(lambda examples: tokenizer(examples['tweet_text'], padding='max_length', truncation=True, max_length=128), batched=True)
tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataloader = DataLoader(tokenized_test['test'], batch_size=16)

# Evaluate the model on test data
model.eval()

# Store predictions and labels
predictions = []
actual_labels = []

# Disable gradient calculations
with torch.no_grad():
    for batch in test_dataloader:
        # Forward pass
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        logits = outputs.logits
        
        # Store predictions and actual labels
        predictions.extend(torch.argmax(logits, dim=1).tolist())
        actual_labels.extend(batch['labels'].tolist())

from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(actual_labels, predictions)
print(f'Accuracy: {accuracy}')

class_report = classification_report(actual_labels, predictions, digits=4)
print("\nClassification Report:\n", class_report)