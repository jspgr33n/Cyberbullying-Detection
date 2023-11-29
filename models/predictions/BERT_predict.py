from transformers import BertForSequenceClassification, BertTokenizer, pipeline
from datasets import load_dataset
import torch

model = BertForSequenceClassification.from_pretrained("../BERT_trained_removed")
tokenizer = BertTokenizer.from_pretrained("../BERT_trained_removed")

nlp = pipeline("text-classification", model=model, tokenizer=tokenizer)

sample_text = "You are a stupid gucking little shit go die in a hole you fuck"
prediction = nlp(sample_text)

print(prediction)


def tokenize_function(examples):
    return tokenizer(examples['tweet_text'], padding='max_length', truncation=True, max_length=128)

file_path = '../../data/test_data_removed.csv' 
dataset = load_dataset('csv', data_files={'test': file_path})

tokenized_test = dataset.map(tokenize_function, batched=True)

tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])

# print(tokenized_test['test'][0])

from torch.utils.data import DataLoader

test_dataloader = DataLoader(tokenized_test['test'], batch_size=16)

model.eval()

predictions = []
actual_labels = []

# Disable gradient calculations
with torch.no_grad():
    for batch in test_dataloader:
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=1).tolist())
        actual_labels.extend(batch['labels'].tolist())

from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(actual_labels, predictions)
print(f'Accuracy: {accuracy}')

class_report = classification_report(actual_labels, predictions)
print("\nClassification Report:\n", class_report)