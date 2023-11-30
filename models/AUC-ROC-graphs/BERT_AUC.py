import torch
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load your model and tokenizer
model = BertForSequenceClassification.from_pretrained("../BERT_trained_removed")
tokenizer = BertTokenizer.from_pretrained("../BERT_trained_removed")

# Load and tokenize your dataset
file_path = '../../data/test_data_removed.csv' 
dataset = load_dataset('csv', data_files={'test': file_path})
tokenized_test = dataset.map(lambda examples: tokenizer(examples['tweet_text'], padding='max_length', truncation=True, max_length=128), batched=True)
tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataloader = DataLoader(tokenized_test['test'], batch_size=16)

# Evaluate the model
model.eval() 
softmax = torch.nn.Softmax(dim=1)

all_probs = []
all_labels = []

with torch.no_grad():
    for batch in test_dataloader:
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        logits = outputs.logits
        probs = softmax(logits)
        all_probs.extend(probs[:, 1].tolist())  # Assuming the positive class is at index 1
        all_labels.extend(batch['labels'].tolist())

# Calculate ROC-AUC
roc_auc = roc_auc_score(all_labels, all_probs)
print(f"ROC-AUC Score: {roc_auc}")

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label='ROC Curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # Random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve BERT')
plt.legend(loc="lower right")
plt.show()
