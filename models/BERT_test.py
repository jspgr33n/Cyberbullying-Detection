from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
print(torch.__version__)
# Your existing code for loading and tokenizing the dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['tweet_text'], padding='max_length', truncation=True, max_length=128)

file_path = '../data/train_data.csv'
dataset = load_dataset('csv', data_files={'train': file_path})

train_valid_split = dataset['train'].train_test_split(test_size=0.2)

train_dataset = train_valid_split['train']
valid_dataset = train_valid_split['test']

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_valid = valid_dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])
tokenized_valid.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])

# Compute class weights
labels = tokenized_train['labels'].numpy()
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

# Custom Loss Function
class CustomLoss(nn.Module):
    def __init__(self, class_weights):
        super(CustomLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, outputs, labels):
        return self.loss(outputs.logits, labels)

# Initialize the custom loss function
custom_loss = CustomLoss(class_weights_tensor)

# Custom Trainer
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(CustomTrainer, self).__init__(*args, **kwargs)
        self.loss_func = CustomLoss(class_weights_tensor)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.loss_func(outputs, labels)
        return (loss, outputs) if return_outputs else loss

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    save_steps=10_000,
    eval_steps=500,
    use_mps_device=True
)

# Initialize the Trainer with custom loss
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
model.save_pretrained("./BERT_test_trained")
tokenizer.save_pretrained("./BERT_test_trained")

# Evaluate the model (if you have a test dataset)
# results = trainer.evaluate(tokenized_test)
