import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset

loaded_base_model = AutoModel.from_pretrained('../base_model')

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
    
loaded_model = CustomClassifier(loaded_base_model, 2)

loaded_model.load_state_dict(torch.load('../custom_classifier_state.pth'))

loaded_tokenizer = AutoTokenizer.from_pretrained('../twihin_trained')

example_text = "I love you"

inputs = loaded_tokenizer(example_text, return_tensors="pt", padding=True, truncation=True, max_length=128)

input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

with torch.no_grad():
    _, logits = loaded_model(input_ids=input_ids, attention_mask=attention_mask)

    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predictions = torch.argmax(probabilities, dim=-1)

print("Predictions:", predictions)
