import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset

# Load the pre-trained model
base_model = AutoModel.from_pretrained('Twitter/twhin-bert-base')

# Create a custom model with a classification head
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

# Specify the number of labels for your classification task
NUM_LABELS = 2  # Adjust according to your dataset

model = CustomClassifier(base_model, num_labels=NUM_LABELS)

# Load and tokenize the dataset
tokenizer = AutoTokenizer.from_pretrained('Twitter/twhin-bert-base')
file_path = '../data/train_data_removed.csv' 
dataset = load_dataset('csv', data_files={'train': file_path})

def tokenize_function(examples):
    return tokenizer(examples['tweet_text'], padding='max_length', truncation=True, max_length=128)

train_valid_split = dataset['train'].train_test_split(test_size=0.2)
train_dataset = train_valid_split['train']
valid_dataset = train_valid_split['test']

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_valid = valid_dataset.map(tokenize_function, batched=True)

# Ensure labels are included in the tokenized datasets
tokenized_train = tokenized_train.map(lambda examples: {'labels': examples['labels']}, batched=True)
tokenized_valid = tokenized_valid.map(lambda examples: {'labels': examples['labels']}, batched=True)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir="results_twhin",
    num_train_epochs= 3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='logs_twhin',
    logging_steps=10,
    evaluation_strategy="steps",
    save_steps=10_000,
    eval_steps=500
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the model and tokenizer
# model.save_pretrained("./twhin_trained")

torch.save(model.state_dict(), 'custom_classifier_state_removed.pth')
model.base_model.save_pretrained('./twhin_trained_removed')
tokenizer.save_pretrained("./twhin_trained_removed")
