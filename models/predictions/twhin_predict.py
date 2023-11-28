import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

loaded_base_model = AutoModel.from_pretrained('../twhin_trained')

class CustomClassifier(nn.Module):
    def __init__(self, base_model, num_labels):
        super(CustomClassifier, self).__init__()
        self.num_labels = num_labels
        self.base_model = base_model
        self.classifier = nn.Linear(base_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        return loss, logits
    
model = CustomClassifier(loaded_base_model, 2)
model = CustomClassifier(AutoModel.from_pretrained('../twhin_trained'), 2)
model.load_state_dict(torch.load('../custom_classifier_state.pth'))
model.eval()

# Load and preprocess the test data
tokenizer = AutoTokenizer.from_pretrained('../twhin_trained')
test_file_path = '../../data/test_data.csv'
test_dataset = load_dataset('csv', data_files={'test': test_file_path})

def tokenize_function(examples):
    return tokenizer(examples['tweet_text'], padding='max_length', truncation=True, max_length=128)

tokenized_test = test_dataset['test'].map(tokenize_function, batched=True)

# Evaluation
# If you have labels in your test data, you can calculate metrics like accuracy
# Here's a simple way to do it using a loop. For larger datasets, consider batch processing.

all_predictions = []
all_true_labels = []

with torch.no_grad():
    for i in range(len(tokenized_test)):
        input_ids = torch.tensor(tokenized_test[i]['input_ids']).unsqueeze(0)  # Add batch dimension
        attention_mask = torch.tensor(tokenized_test[i]['attention_mask']).unsqueeze(0)
        
        outputs = model(input_ids, attention_mask)
        logits = outputs[1]
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        
        all_predictions.extend(predictions)
        all_true_labels.extend([tokenized_test[i]['labels']])

# Convert to numpy arrays for compatibility with scikit-learn
all_predictions = np.array(all_predictions)
all_true_labels = np.array(all_true_labels)

# Generate the classification report
report = classification_report(all_true_labels, all_predictions, target_names=['Class 0', 'Class 1'])  # Adjust target names as per your classes
print(report)